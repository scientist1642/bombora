# Reads db logs and visualizes on visdom

import time
import base64
import numpy as np
import math
import subprocess
import os.path
import getpass
from visdom import Visdom
import sys
import signal

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from jinja2 import Template

from utils import dblogging
from utils.misc import human_format

class Mytemplates:
    List =  Template('''
          <ul>
            {% for n in xs %}
                <li><strong>{{n}}</strong></li>
            {% endfor %}
        </ul>
            ''')
    
    Videos =  Template('''
            {% for data in xs %}
                <video controls>
                    <source type="video/{{ext}}" src="data:video/{{ext}};base64,{{data}}">
                    Try Chrome
                </video>

            {% endfor %}
        </ul>
        ''')


class Dashboard:
    '''Builds LIVE dashboard of visdom based on sqlite log files
       instruction: Run visdom server and then run this script.
       Protocol V1 of dblogger
    '''

    def __init__(self, dbdir='dblogs', runlist=None, 
            interval = 10):
        '''
        dbdir: specifies where to look for sqlite log files,
        runlist: list of run names to visualize, i.e.(07.04-04.53(no_descr))
                  default is to make a seperate env for all of them.
        interval: time interval to update dashboard
        '''
        self.dbdir = dbdir
        self.runlist = runlist
        self.interval = interval
        
        # go through each requested env folder, find all sqlite files, take last one
        if self.runlist is None:
            # find all sqlite files, and add latest [1]created db TODO add option
            self.runlist = []
            for env_name in ['Seaquest-v0', 'Breakout-v0']:
                tmp = []
                envdbdir = os.path.join(dbdir, env_name)
                for f in os.listdir(envdbdir):
                    if f.endswith(".sqlite3"):
                        fullpath = os.path.join(dbdir, env_name, f)
                        tmp.append((os.path.getctime(fullpath), fullpath, f))
                latestrun = max(tmp)[1]
                self.runlist.append((f, fullpath)) # run name and path to db

        print('Detected following db logs')
        for name, fullpath in self.runlist:
            print ('name : {}, path: {}'.format(name, fullpath))
        print('=============================')
   
    def _update_env(self, db, viz, wins):
        ''' update visdom for specific env,

            db: DBreader object
            viz: visdom env
            windows: dict of windows on this env
        '''
        
        # now db is an iterator which might be exhausted previously,
        # but new logs might have been added, continue consuming

        for data in db: # data is one of named tuple in dblogger
            if isinstance(data, dblogging.ExperimentArgs):
                self._plot_args(data, db, viz, wins)
            elif isinstance(data, dblogging.TestSimple):
                self._plot_simple_test(data, db, viz, wins)
            elif isinstance(data, dblogging.TestHeavy):
                self._plot_heavy_test(data, db, viz, wins)
            else:
                print ('Unknown tuple instance {}'.format(type(data).__name__))

    def _plot_args(self, data, db, viz, wins):
        arglist = []
        for k, v in data.args.items():
            if k not in ['temp_dir', 'tboard_log_dir', 'db_path']: # we can filter out some keys
                arglist.append(str(k) +' : '+ str(v))
        viz.text(Mytemplates.List.render(xs=arglist), wins['runinfo'], 
                opts={'title': 'Arguments Info'})

    def _plot_simple_test(self, data, db, viz, wins):
        # =================== Average Reward Plot ============== 
        print ('Starting simple plot')
        x = np.array([data.glsteps])
        y = np.array([data.avgscore])
        if not 'scores' in wins:
            #TODO also plot std
            win = viz.line(X=x, Y=y,opts={'title':'Average Score'})
            wins['scores'] = win
        else:
            viz.updateTrace(X=x,Y=y,win=wins['scores'])
        
        print ('Done simple plot')



    def _plot_heavy_test(self, data, db, viz, wins):
        print('Started heavy plot')
        step = human_format(data.glsteps)
        video_title = 'Step: {}, Score: {}'.format(step,data.score)
        #viz.video(videofile=data.video, ispath=False, extension='mp4', 
        #        opts={'title':video_title})
        
        # Get real video coming from gym monitor
        real_video = base64.b64encode(data.video).decode('utf8')
        real_video_tag = Mytemplates.Videos.render(xs=[real_video], ext='mp4')
        
        state_frames=np.moveaxis(data.states, 1, 3).squeeze()
        randconv_frames=data.randomconv
        predvalues = data.predvalues.squeeze()
        action_distr = data.action_distr
        
        
        # create figs axis and some fine tuning
        fig,((ax4,ax2),(ax3,ax1)) = plt.subplots(2, 2, figsize=(6, 6), dpi=80)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)
        #ax2.yaxis.tick_right()
        #ax1.grid(False)
        #ax2.grid(False)
        ax2.set_ylim([min(predvalues)-0.2, max(predvalues)]) 
        
        # animate things
        def update(num, frames, convs, predvalues, rects, convimg, 
                stateimg, predline, action_distr):
            # update observation images plot 1
            stateimg.set_array(frames[num])
            # update random conv visualization 
            convimg.set_array(convs[num])
            # update predicted value estimates plot 2
            hist = min(num, 50)
            predline.set_data(np.linspace(0, 1, hist), predvalues[num-hist:num])
            # update  action distribution plot 3
            #import ipdb; ipdb.set_trace()
            for rect, h in zip(rects, action_distr[num]):
                rect.set_height(h)

            return (stateimg, convimg, predline)
        predline = matplotlib.lines.Line2D([],[], color='red')
        ax2.add_line(predline)
        stateimg = ax1.imshow(state_frames[0],animated=True)
        convimg = ax4.imshow(randconv_frames[0],animated=True, cmap='gray')
        num_actions = action_distr.shape[1]
        rects = ax3.bar(range(num_actions), [0]*num_actions) #align='center'
        
        TO_RENDER= min(800, state_frames.shape[0])
        ani = animation.FuncAnimation(fig, update, TO_RENDER, 
                fargs=(state_frames,randconv_frames, predvalues, rects, 
                    stateimg, convimg, predline,
                    action_distr), interval=50, blit=True)
        
        # conver to video
        state_video_tag = ani.to_html5_video()
        
        viz.text(real_video_tag, opts={'title':video_title})
        viz.text(state_video_tag, opts={'title':video_title})
        print ('Done heavy plot')


    def update_envs(self):
        ''' update all visdom envs '''
        for db, viz, wins in self.db_env_wins:
            self._update_env(db, viz, wins)

    def start(self):
        # make list of db, vizs, windows triplets
        self.db_env_wins = []
        for (runname,runpath) in self.runlist:
            db = dblogging.DBReader(os.path.join(runpath))
            viz = Visdom(env = runname)
            
            # setup windows in the env
            wins = {'runinfo': viz.text('info')}
            
            self.db_env_wins.append((db, viz, wins))

        while True:
            self.update_envs()
            time.sleep(self.interval)


if __name__ == '__main__':
    def preexec_function():
        # Ignore the SIGINT signal by setting the handler to the standard
        # signal handler SIG_IGN.
        # and attach session id to parent process
        #signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.setsid()
    try:
        # run visdom as a subprocess,  
        # should be carefull not to be left in the wild
        prog = subprocess.Popen('python -m visdom.server', shell=True, preexec_fn = preexec_function,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #prog = subprocess.Popen('python -m visdom.server', shell=True, preexec_fn = preexec_function)
        time.sleep(1)
        if len(sys.argv) != 1:
            # running server
            dbdir = sys.argv[1]
            #runlist = [sys.argv[1]]
            runlist = None
        else:
            dbdir = 'dblogs'
            runlist = None

        dashboard = Dashboard(dbdir=dbdir)
        dashboard.start()
    except KeyboardInterrupt:
        print ('keyInterrupted')
    finally:
        os.killpg(os.getpgid(prog.pid), signal.SIGTERM)
        #sys.exit(0)

