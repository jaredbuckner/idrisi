import idrisi.heightmap as heightmap
import idrisi.jrandom as jrandom
import idrisi.jutil as jutil
import queue
import threading

class MapWorker(threading.Thread):
    def __init__(self, quitevent):
        super().__init__(sarget=self._work)
        self._map = None
        self._jr = jrandom.JRandom()
        self._vp = None

        self._inqueue = queue.SimpleQueue()
        self._quitevent = quitevent

    def _work(self):
        while(not self._quitevent.is_set()):
            try:
                callback, outqueue = self._inqueue.get(timeout=0.2)
                try:
                    results = callback(self._map, self._jr, self._vp)
                    if(outqueue is not None):
                        outqueue.put(("?", results))
                except BaseException as e:
                    outqueue(("!", e))
                    pass
                pass
                
            except queue.Empty:
                pass

    def enqueue_work(self, callback, outqueue=None):
        self._inqueue.put((callback, outqueue))

    def set_viewport(self, *args, *kwargs):
        def _set_viewport_callback(hmap, jr, vp):
            ## Work on more of this later...
            
        
            
