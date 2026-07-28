import imageio

class RLVideoRecorder:
    def __init__(self, env, name="biem"):
        self.env = env
        self.frames = []
        self.name = name
    
    def __get_frame(self):
        if hasattr(self.env, "env"):
            return self.env.env.render(mode="rgb_array")
        elif hasattr(self.env, "_env"):
            #try:
            return self.env._env.render(mode="rgb_array")
            #except:
            #print("##### Not possible to get frame for some reason. See rl_video_recorder.py file ######")
        elif hasattr(self.env, "original_env"):
            return self.env.original_env.render()

        return None
    
    def record_video(self):
        #frame = self.__get_frame()
        frame = self.env.get_rgb_frame()
        if frame is not None:
            self.frames.append(frame)
    
    def save_video(self, index):
        imageio.mimwrite(f"videos/{self.name}_episode_{index}.mp4", self.frames, format='FFMPEG', fps=10)
    
    def reset_frames_buffer(self):
        self.frames = []
        
        
