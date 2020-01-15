''' Program to keep track of the progress of a task '''

import time


class ETA():
    ''' contains methods to compute Excpected Time of Arrival and
    task completion percentage '''

    def __init__(self,max_value) :
        self.max_value = max_value
        self.start_time = 0
        self.curr_time = 0

    def start(self) :
        self.start_time = time.time()
        completion = 0
        eta = 'Null'
        return str(completion), eta

    def update(self, value) :
        if value == 0 :
            completion = 0
            eta = 'Null'
            return str(completion), eta

        self.curr_time = time.time()
        dt = (self.curr_time - self.start_time)
        completion = (value / self.max_value) * 100
        rem_time = (dt/value) * self.max_value - dt # Remaining time in seconds
        days = int(rem_time // (24 * 3600))
        rem_time = rem_time % (24 * 3600)
        hours = int(rem_time // 3600)
        rem_time %= 3600
        minutes = int(rem_time // 60)
        rem_time %= 60
        seconds = int(rem_time)
        eta = str(days) + ' days ' + str(hours) + ':' + str(minutes) + ':' + str(seconds)

        return str(completion), eta

    def finish(self) :
        completion = 100
        self.curr_time = time.time()
        dt = (self.curr_time - self.start_time)
        days = int(dt // (24 * 3600))
        dt = dt % (24 * 3600)
        hours = int(dt // 3600)
        dt %= 3600
        minutes = int(dt // 60)
        dt %= 60
        seconds = int(dt)
        eta = str(days) + ' days ' + str(hours) + ':' + str(minutes) + ':' + str(seconds)
        return str(completion), eta
