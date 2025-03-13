import time
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from IPython.display import clear_output

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 
        'max_steps': ep_length, 'print_rewards': True, 'save_video': False, 
        'fast_video': True, 'session_path': sess_path, 'gb_path': '../PokemonRed.gb', 
        'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
    }
    
    num_cpu = 1
    env = make_env(0, env_config)()

    file_name = 'session_4da05e87_main_good/poke_439746560_steps'
    
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        
    obs, info = env.reset()
    while True:
        action = 7  # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Render and display at 5 FPS
        frame = np.array(env.render(reduce_res=False, add_memory=True, update_mem=True))

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            clear_output(wait=True)  # Clear previous frame to avoid clutter
            #time.sleep(1)
            cv2_imshow(frame)  # Display frame

        # Add a delay 
        time.sleep(2 / 1)

        if truncated:
            break
            
    env.close()
