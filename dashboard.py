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
import logging
import argparse

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from jinja2 import Template
from utils import dblogging
from utils.misc import human_format

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(message)s') # include timestamp



parser = argparse.ArgumentParser(description='Dashboard')
parser.add_argument('--env', metavar='ENV',
                    help='environment to run visualization')
parser.add_argument('--dbdir', metavar='DBDir',
                    help='db dir, for local case dblogs')
parser.add_argument('--heavy-ids', nargs='+', type=int, default=[],
                    help='idds of heavy rendering')



class Mytemplates:
    List =  Template('''
          <ul>
            {% for n in xs %}
                <li><strong>{{n}}</strong></li>
            {% endfor %}
        </ul>
            ''')
    
    Videos_bytes =  Template('''
            {% for data in xs %}
                <video controls width="{{width}}" height="{{height}}">
                    <source type="video/{{ext}}" src="data:video/{{ext}};base64,{{data}}">
                    Try Firefox or Chrome
                </video>

            {% endfor %}
        </ul>
        ''')

    Videos =  Template('''
            {% for path in xs %}
                <video controls width="{{width}}" height="{{height}}">
                    <source src="static/{{path}}" type="video/{{ext}}" >
                    Try Firefox or Chrome
                </video>

            {% endfor %}
        </ul>
        ''')


def _update_env(data, cache, viz, wins, heavy_ids):
    ''' update visdom for specific env,
        
        viz: visdom env
        cache: directory path to save videos
        windows: dict of windows on this env
        
    '''
    evtname = data['evtname'] 
    if evtname == 'ExperimentArgs':
        _plot_args(data, cache, viz, wins)
    elif evtname =='SimpleTest':
        _plot_simple_test(data, cache, viz, wins, )
    elif evtname == 'HeavyTest':
        _plot_heavy_test(data, cache, viz, wins, heavy_ids)
    else:
        logging.warning('Unknown tuple instance {}'.format(type(data).__name__))

def _plot_args(data, cache,  viz, wins):
    def get_pr(item):
        k, v= item
        pr = {'source_code': 0}
        if k in pr:
            return pr[k]
        else:
            return 100
    arglist = sorted(data['args'].items(), key=get_pr)

    xs = []
    for k, v in arglist:
        if k not in ['temp_dir', 'tboard_log_dir', 'db_path']: # we can filter out some keys
            kk, vv = str(k), str(v)
            if kk == 'source_code':
                vv =  '<a href="{}">code</a>'.format(vv)
            xs.append(kk +' : '+ vv)
    viz.text(Mytemplates.List.render(xs=xs), wins['runinfo'], 
            opts={'title': 'Arguments Info'})

def _plot_simple_test(data, cache, viz, wins):
    # =================== Average Reward Plot ============== 
    #logging.info('Starting simple plot')
    x = np.array([data['glsteps']])
    y = np.array([data['avgscore']])
    if not 'scores' in wins:
        #TODO also plot std
        win = viz.line(X=x, Y=y,opts={'title':'Average Score', 
            'height':274, 'markersize':1})
        wins['scores'] = win
    else:
        viz.updateTrace(X=x,Y=y,win=wins['scores'])
    
    #logging.info('Done simple plot')

def render_agent_video(data, cache):
    ''' renders an agent video and retunrs a path to it '''
    video_name = 'agent-{}.mp4'.format(data['glsteps'])
    video_path = os.path.join(cache, video_name)
    if os.path.isfile(video_path):
        # return cached version
        return video_path

    writer = animation.writers['ffmpeg']
    writer = writer(fps=15, metadata=dict(artist='me'), bitrate=1800)

    state_frames=np.moveaxis(data['states'], 1, 3).squeeze()
    randconv_frames=data['randomconv']
    predvalues = data['predvalues'].squeeze()
    action_distr = data['action_distr']
    
    
    # create figs axis and some fine tuning
    fig,((ax4,ax2),(ax3,ax1)) = plt.subplots(2, 2, figsize=(6, 6), dpi=80)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.01, hspace=0.01)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    #ax2.yaxis.tick_right()
    #ax1.grid(False)
    #ax2.grid(False)
    ax2.set_ylim([min(predvalues)-0.2, max(predvalues)]) 
    ax3.set_ylim(0, 1) 
    
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
    time_start_render = time.time()
    state_video_tag = ani.to_html5_video(width=242, height=274)
    logging.info('Rendering time {}'.format(time.time() - time_start_render))
    plt.close()
    
    ani.save(video_path, writer=writer)
    return video_path
    
def render_real_video(data, cache):
    video_name = 'real-{}.mp4'.format(data['glsteps'])
    video_path = os.path.join(cache, video_name)
    if os.path.isfile(video_path):
        return video_path
    
    with open(video_path, 'wb') as f:
        f.write(data['video'])
    #real_video = base64.b64encode(data.video).decode('utf8')
    #real_video_tag = Mytemplates.Videos.render(xs=[real_video], ext='mp4', 
    #        width=242, height=274)
    return video_path

def _plot_heavy_test(data, cache, viz, wins, heavy_ids):
    logging.info('Started heavy plot')
    step = human_format(data['glsteps'])
    video_title = 'Step: {}, Score: {} ID: {}'.format(
            step, data['score'],data['idd'])
    #viz.video(videofile=data.video, ispath=False, extension='mp4', 
    #        opts={'title':video_title})
    
    
    
    # Get real video coming from gym monitor
    #real_video = base64.b64encode(data.video).decode('utf8')
    real_video = render_real_video(data, cache) 
    real_video_tag = Mytemplates.Videos.render(xs=[real_video], ext='mp4', 
            width=242, height=274)
    
    if data['idd'] in heavy_ids:
        agent_video = render_agent_video(data, cache)
        agent_video_tag = Mytemplates.Videos.render(xs=[agent_video], ext='mp4', 
                width=242, height=274)
    else:
        agent_video_tag=''

    viz.text(real_video_tag + agent_video_tag, opts={'title':video_title})
    #viz.text(state_video_tag, opts={'title':video_title})
    print ('Done heavy plot')

class Dashboard:
    '''Builds LIVE dashboard of visdom based on sqlite log files
       instruction: Run visdom server and then run this script.
       Protocol V1 of dblogger
    '''

    def __init__(self, dbdir, envname, args, names=[], cachedir='cache',
            interval = 1):
        '''
        dbdir: specifies where to look for sqlite log files,
        env_name: name of the environment i.e. Pong-v0 all of them.
        runnames: list of runnnames i.e. nod-0804-0558
        cachedir: dir to cache renered videos, etc..
        interval: time interval to update dashboard
        
        NOTE, if you want to use caching make symlink of cache in
        visdom/static directory
        '''
        self.dbdir = dbdir
        self.runlist = []
        self.interval = interval
        self.args = args
        
        # go through each requested env folder, find all sqlite files, take last one
        if len(names) == 0:
            # find all sqlite file in env_name and add names of last 2 of them
            tmp = []
            envdbdir = os.path.join(dbdir, envname)
            for name in os.listdir(envdbdir):
                dbpath = os.path.join(dbdir, envname, name)
                if name.endswith(".sqlite3"):
                    without_ext = os.path.splitext(name)[0]
                    tmp.append((os.path.getctime(dbpath), without_ext))
            names = [ x[1] for x in sorted(tmp, reverse=True)[:2] ]
             
        for name in names:
            dbpath = os.path.join(dbdir, envname, name +'.sqlite3')
            cachepath =  os.path.join(cachedir, envname, name)
            #cachepath = os.path.abspath(cachepath)
            if not os.path.exists(cachepath):
                os.makedirs(cachepath)
            self.runlist.append((name, dbpath, cachepath)) 
        
        logging.info('Detected following db logs')
        for name, dbpath, cachepath in self.runlist:
            logging.info ('name : {}, path: {}'.format(name, dbpath))
        logging.info('=============================')


    def update_envs(self):
        ''' update all visdom envs '''
        #pool = multiprocessing.Pool(3)
        #pool.starmap(_update_env, self.tabs) 
        # self.tabs contains db, cache, viz wins
        updated = False
        for db, cache, viz, wins in self.tabs:
            try:
                idd, evtname, data, timestamp = next(db)
                data['idd'] = idd
                _update_env(data, cache, viz, wins, self.args.heavy_ids)
                updated = True
            except StopIteration:
                pass
        return updated

    def start(self):
        # each tab corrresponds to separate log
        self.tabs = []
        for (runname, dbpath, cachepath) in self.runlist:
            db = dblogging.DBReader(dbpath)
            viz = Visdom(env = runname)
            
            # setup windows in the env
            wins = {'runinfo': viz.text('info')}
            
            self.tabs.append((db, cachepath, viz, wins))

        while True:
            updated = self.update_envs()
            if not updated:
                time.sleep(self.interval)


if __name__ == '__main__':
    args = parser.parse_args()
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
        # TODO use argparse
        #dbdir = sys.argv[1]
        #envname = sys.argv[2]

        dashboard = Dashboard(args.dbdir, args.env, args=args)
        dashboard.start()
    except KeyboardInterrupt:
        print ('keyInterrupted')
    finally:
        os.killpg(os.getpgid(prog.pid), signal.SIGTERM)
        #sys.exit(0)

