import pynput
import time
import numpy as np
class pid:
    def __init__(self,p,i,d) -> None:
        self.kp=p
        self.ki=i
        self.kd=d
        self.exp_val=0
        self.now_error=0
        self.last_error=0
        self.prev_error=0
        self.sum_error=0
        self.out=0

    def pid_place(self,exp_val,now_val):
        self.exp_val=exp_val
        self.now_error=exp_val-now_val
        self.sum_error+=self.now_error
        self.out=self.kp *self.now_error +self.ki * self.sum_error +self.kd*(self.now_error-self.last_error)
        self.last_error=self.now_error
        return self.out

    def pid_inc(self,exp_val,now_val):
        self.exp_val=exp_val
        self.now_error=exp_val-now_val
        self.out+=(self.kp*(self.now_error-self.last_error) +self.ki*self.now_error+self.kd*(self.now_error-2*self.last_error+self.prev_error))
        self.prev_error=self.last_error
        self.last_error=self.now_error
        return self.out

if __name__=='__main__':
    ctr = pynput.mouse.Controller()
    mousepid=pid(0.5,0.05,0)
    target_posi=np.array((500,200))
    while True:
        ctr.position=(tuple(mousepid.pid_inc(target_posi,np.array(ctr.position))+ctr.position))
        print(ctr.position[0])
        time.sleep(0.01)
        # if ctr.position==tuple(target_posi):
        #    break