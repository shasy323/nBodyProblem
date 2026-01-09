
####################################################################
# Imports and given constants and classes to be used in your code
# DO NOT CHANGE ANY OF THE BELOW
####################################################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle 
import math

# UNITS USED
# Time: Astronomical Units (AU). 1AU = distance between Sun and earth
# Mass: Solar Mass (M_sun). 1M_sun = mass of the Sun
# Time: Year (yr). 1yr = 1 year (one rotation of the Earth around the Sun)
# Luminosity: Solar Luminosity (L_sun). 1L_sun = luminosity of the Sun
# Velocity: AU/yr

G = 4*(math.pi)**2  # Universal gravitational constant (AU^3/M_sun/yr^2)
scoeff = 22.62      # Scorching coefficient, approx. 0.745 AU from the Sun (L_sun/AU^2)
fcoeff = 3.14       # Freezing coefficient, approx. 2 AU from the Sun (L_sun/AU^2)

class Box:
    def __init__(self,x0,y0,x1,y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.maxSide = max(self.x1-self.x0,self.y1-self.y0)
        self.mx = (self.x0+self.x1)/2
        self.my = (self.y0+self.y1)/2
        
    def isIn(self,p):
        return self.x0 <= p.x <= self.x1 and self.y0 <= p.y <= self.y1
    
    def asTuple(self):
        return (self.x0,self.y0,self.x1,self.y1,self.mx,self.my)
    
    def split4(self):
        bNE = Box(self.mx,self.my,self.x1,self.y1)
        bNW = Box(self.x0,self.my,self.mx,self.y1)
        bSW = Box(self.x0,self.y0,self.mx,self.my)
        bSE = Box(self.mx,self.y0,self.x1,self.my)
        return (bNE,bNW,bSW,bSE)

    # Build a box that encloses all Bodies from an array 
    @staticmethod
    def getBox(P):
        if len(P) == 0: return None
        x0 = x1 = P[0].x
        y0 = y1 = P[0].y
        for p in P: 
            if p.x < x0: x0 = p.x
            if p.y < y0: y0 = p.y
            if p.x > x1: x1 = p.x
            if p.y > y1: y1 = p.y
        return Box(x0,y0,x1,y1)    
    
    def __str__(self):
        return f"Box({self.x0},{self.y0},{self.x1},{self.y1})"
        
class GNode:
    def __init__(self,box):
        self.box = box
        self.COM = None           # this node’s COM
        self.nbodies = 0          # number of bodies contained in node
        self.p = None             # if this node is a leaf with a Body
        self.children = None      # children are: [NE, NW, SW, SE]
        self.updateCOM()

    def isLeaf(self):
        return self.nbodies < 2
        
    def updateCOM(self):
        if self.isLeaf(): 
            if self.p == None:
                self.COM = Body(0, self.box.mx, self.box.my)
            else:
                self.COM = Body(self.p.m, self.p.x, self.p.y)
            return
        x = y = m = 0
        for c in self.children:
            x += c.COM.x*c.COM.m
            y += c.COM.y*c.COM.m
            m += c.COM.m
        self.COM = Body(m, x/m, y/m)

    def niceStr(self): 
        S = ("├","─","└","│")
        angle = S[2]+S[1]+" "
        vdash = S[0]+S[1]+" "
        
        def niceRec(ptr,acc,pre,A):
            if ptr == None: raise Exception("A None GNode was found")
            val = f"{len(A)}:{ptr.box},{ptr.nbodies}"
            A.append(f"({ptr.COM.m}, {ptr.COM.x}, {ptr.COM.y})")
            if ptr.children==None: return acc+pre+val
            if pre == vdash: pre2 = S[3]+"  "
            elif pre == angle: pre2 = "   "
            else: pre2 = ""
            T = [vdash,vdash,vdash,angle]
            for i in range(4):
                T[i] = niceRec(ptr.children[i],acc+pre2,T[i],A)
            return acc+pre+val+"\n"+T[0]+"\n"+T[1]+"\n"+T[2]+"\n"+T[3]
            
        A = []
        s = niceRec(self,"","",A)+"\n"
        for i in range(len(A)):
            s += f"\n{i}{' '*(3-len(str(i)))}-> {A[i]}"
        return s
    
class Stack:
    def __init__(self):
        self.inList = []
        
    def push(self,v):
        self.inList.append(v)

    def pop(self):
        if len(self.inList) == 0: raise Exception("Popping from an empty stack")
        return self.inList.pop()
    
    def isEmpty(self):
        return len(self.inList) == 0
    
    def size(self):
        return len(self.inList)

    def toArray(self):
        return self.inList
    
    def __str__(self):
        return str(self.inList)
####################################################################
# Question 1: Intro to the n-body problem
# To solve: closest distance, predict good eras in 3-body problem
####################################################################

class Body:
    def __init__(self,m,x,y,vx=0,vy=0):
        self.m = m     # mass
        self.x = x     # position
        self.y = y     # position
        self.vx = vx   # velocity
        self.vy = vy   # velocity
        
    def squareDist(self,other):
        return (self.x-other.x)**2+(self.y-other.y)**2

    def __str__(self):
        return f"Body({self.m},{self.x},{self.y},{self.vx},{self.vy})"    

    def __repr__(self):
        return str(self)    

    def asTuple(self):
        return (self.m,self.x,self.y,self.vx,self.vy)    
    
    # Gives the next position and velocity of the current Body to its position
    # and velocity after time dt las elapsed, taking into account the pull
    # forces from the bodies in the array Bodies.
    def next(self, Bodies, dt):
        ret = Body(self.m, self.x, self.y)
        ax = ay = 0       
        # for each p in Bodies we compute their force on this Body and add it to its acceleration 
        for p in Bodies:
            if p == self: continue # current Body does not affect itself          
            # euclidian distance between p and this Body
            sq_distance = self.squareDist(p)
            # see e.g. https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation
            # for the vector form of Newton's law of gravity            
            ax += (p.x - self.x) *  p.m * G / (sq_distance**1.5)
            ay += (p.y - self.y) *  p.m * G / (sq_distance**1.5)
        # we compute displacement due to acceleration that we computed above
        # and due to its current speed because of inertia        
        ret.x += dt*dt*ax + dt*self.vx
        ret.y += dt*dt*ay + dt*self.vy
        # compute the velocity vectors
        ret.vx = (ret.x-self.x)/dt
        ret.vy = (ret.y-self.y)/dt

        return ret

    # Predict years of stability that self planet is going to have in a 3-sun solar system.
    # It suffices to take into account just the three suns (sunA,sunB,sunC) and the planet.
    # Stability is broken if one of the following criteria is violated: 
    # a. No scorching: lA/dA + lB/dB + lC/dC must be less than scoeff
    # b. No freezing: lA/dA + lB/dB + lC/dC must be greater than fcoeff
    # Notes: 
    # - dA, dB, dC are the squared distances between sunA, sunB, sunC and self, and 
    # - lA, lB, lC are the luminosities of sunA, sunB, sunC 
    # respectively. 
    def threeBodyProblem(self,sunA,sunB,sunC,lA,lB,lC): # 15%
        sunLuminosities = [(sunA, lA), (sunB, lB), (sunC, lC)]
        s = Simulation([sunA, sunB, sunC, self])
        sim = s.run()
        bodyIndex = 3

        for time, dt in enumerate(sim):
            totalLuminosity = 0
            body = dt[bodyIndex]
            for sun in range(0, bodyIndex):
                curSunLum = sunLuminosities[sun][1]
                totalLuminosity += (curSunLum / body.squareDist(dt[sun]))
            if totalLuminosity > scoeff or totalLuminosity < fcoeff:
                return time * s.dt
        return 10.01
            
    
class Simulation:
    # Default simulation time 10yr, step time 0.01yr
    def __init__(self, Bodies, total_time = 10, dt=0.01):
        self.bodies = Bodies
        self.total_time = total_time
        self.dt = dt
        self.timesteps = int(total_time/dt)
        
    # Runs the simulation and produces an array of arrays of Bodies.
    # The t-th entry in the list is the position of Bodies after the t-th timestep
    # has been simulated    
    def run(self):
        pss = [None]*(self.timesteps+1)
        pss[0] = self.bodies
        for t in range(self.timesteps):
            # for every Body in the current timestep add its next position in next timestep
            pss[t+1] = [pss[t][i].next(pss[t],self.dt) for i in range(len(self.bodies))]
        return pss


    # Gets the closest Distance between through all time intervals between two planets
    # Currently O(n^3) ((Yikes))
    def closestDistance(self): # 15% 
        if len(self.bodies) < 2: # Checking whether there is less than 2 bodies in sim
            return None
        sim = self.run()
        minDistance = -1
        for dt in sim:
            for body in range(len(dt)):
                for otherBody in range(body + 1, len(dt)):
                    if minDistance == -1:
                        minDistance = dt[body].squareDist(dt[otherBody])
                    else:
                        minDistance = min(minDistance, dt[body].squareDist(dt[otherBody]))
        return math.sqrt(minDistance)     
        
    
    # Shows an animation of the simulation using matplotlib
    def show(self,x0,y0,x1,y1):
        pss = self.run()
        # get figure and axes objects
        fig, ax = plt.subplots()
        # set some reasonable zoom on axes
        ax.set_xlim(x0,x1)
        ax.set_ylim(y0,y1)
        # put labels on the window for Bodies with different colours
        scatter = []
        for i in range(len(self.bodies)):
            scatter.append(ax.scatter([], [], marker='o', label=f'Body {i}'))
         # add timestep text to the legend
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        # update function from one position to the next
        def update(frame):
            for i in range(len(self.bodies)):
                scatter[i].set_offsets([pss[frame][i].x, pss[frame][i].y])
            time_text.set_text(f'Timestep: {frame}')                
            return scatter + [time_text]
        # generate and show the animation
        a = FuncAnimation(fig, update, frames=len(pss), interval=1, blit=True, repeat=False)
        plt.xlabel("X coordinate (AU)")
        plt.ylabel("Y coordinate (AU)")
        plt.title("Celestial body trajectories")
        plt.legend()
        plt.show()    

####################################################################
# Question 2: The tree Gadget
# To solve: add and remove elements on a Gadget
####################################################################

class Gadget:
    def __init__(self,box):
        self.root = GNode(box)
        self.size = 0
        
    # Add Body p to this Gadget [30% TODO]        
    def add(self, p):
        node = self.root
        self.size += 1 #update once
        path = [] #array to store node paths
        added = False

        while added == False: 
            node.nbodies += 1 
            path = append(path, node)
            
            if node.children is not None: #Internal
                
                for child in node.children:
                    if child.box.isIn(p):
                        node = child #Find
                        break
                continue 
                
            if node.p is None: 
                node.p = p 
                i = len(path) - 1
                while i >= 0:
                    path[i].updateCOM()
                    i -= 1
                added = True 
                continue
                
            temp = node.p 
            node.p = None 
            node.children = [GNode(b) for b in node.box.split4()] 
            
            for child in node.children:
                if child.box.isIn(temp):
                    child.p = temp
                    child.nbodies = 1
                    child.updateCOM()
                    break  
        
            for child in node.children:
                if child.box.isIn(p):
                    node.updateCOM()
                    node = child
                    break 

    def remove(self, p):
        root = self.root
        if not (root.box.isIn(p)):
            return
        if (self.removeRec(p, self.root, 0)):
            self.size -= 1
            print("SUCCESS")
        else:
            raise Exception("ahh")
        return 

    def removeRec(self, p, gnode, n): # n is for testing
        if (gnode.children == None): # If leaf node (could be replaced with isLeaf)
            if (gnode.p == p): 
                # If p does exist
                gnode.p = None
                gnode.nbodies = 0
                gnode.updateCOM()
                return True
                
            return False # p does not exist
                
        else: # If not leaf node  
            removed = False # Keeps track if p successfully removed
            for i in range(len(gnode.children)):
                if gnode.children[i].box.isIn(p):
                    removed = self.removeRec(p, gnode.children[i], n+1) 
                    if (removed and gnode.children[i] == p): # To remove p from children
                        gnode.children[i] == None
                    break # For bodies on a boundary
                    
            if (removed): # To update if a removal has occurred 
                gnode.nbodies -= 1
    
                if (gnode.nbodies <= 1): # incase of false internal
                    for i in gnode.children: # Raises the child node up
                        if i.p != None:
                            gnode.p = i.p

                    gnode.children = None
                gnode.updateCOM()
                
            return removed
            
        print("ERROR - NO VALUE RETURNED FROM RETURNREC") # Should never be reached
        return 
            
    ''' Old remove function
    # Remove Body p from this Gadget [20% TODO]
    def remove2(self,p):
        found = False
        stack = Stack()
        stack.push(self.root)
        while not stack.isEmpty():
            n = stack.pop()
            if n.isLeaf():
                continue
            else:
                for i in range(len(n.children)):
                    if n.children[i].p == p:
                        found = True
                        n.children[i] = GNode(n.children[i].box)
                        n.nbodies -= 1
                        self.size -= 1
                        Gadget.checkInternal(self.root)
                        break
                        
                    stack.push(n.children[i])

    # Helper function to remove false internals
    @staticmethod
    def checkInternal(node):
        if node.children is not None:
            for c in node.children:
                Gadget.checkInternal(c)

        if node.isLeaf() and node.children is None:
            return
        
        if node.isLeaf() and node.children is not None:
            for c in node.children:
                if c.p is not None:
                    node.p = c.p
                    break
            node.children = None
            node.updateCOM()
    '''
        
    def __str__(self):
        return self.root.niceStr()

    # Collect all Bodies from this Gadget and return them in an array
    def getBodies(self):
        A = [None]*self.size
        i = 0; stack = Stack(); stack.push(self.root)
        while not stack.isEmpty():
            n = stack.pop()
            if n.isLeaf():
                if n.p is not None:
                    A[i] = n.p
                    i += 1
            else:
                for c in n.children: stack.push(c)
        return A   

    # Build a new Gadget and add in it all Bodies from array ps
    @staticmethod
    def fromBodies(ps):
        # calculate bottom-left and top-right positions
        if ps == []: return None
        x0 = x1 = ps[0].x
        y0 = y1 = ps[0].y
        for p in ps:
            if p.x < x0: x0 = p.x
            elif p.x > x1: x1 = p.x
            if p.y < y0: y0 = p.y
            elif p.y > y1: y1 = p.y
        # build gadget and add bodies 
        g = Gadget(Box(x0,y0,x1,y1))
        for p in ps: g.add(p)
        return g    
    
    @staticmethod
    def _drawNode(ax, node, showCom):
        x0, y0, x1, y1, mx, my = node.box.asTuple()
        w = x1-x0; h = y1-y0
        # draw the node rectangle
        rect = Rectangle((x0, y0), w, h, fill=False, linewidth=0.8, alpha=0.5)
        ax.add_patch(rect)
        # draw COM for internal nodes
        if showCom and not node.isLeaf():
            ax.plot([node.COM.x], [node.COM.y], marker='+', markersize=6)
            ax.plot([mx, node.COM.x], [my, node.COM.y], linewidth=0.3)            
        # draw leaf contents
        if node.isLeaf(): 
            if node.p is not None:
                ax.plot([node.p.x], [node.p.y], marker='o', markersize=3)
        else:
            # recurse into children [NE, NW, SW, SE]
            for c in node.children:
                if c is not None:
                    Gadget._drawNode(ax, c, showCom)
                
    def plot(self, figsize=(6,6)): # figsize : (w,h) in inches
        showCom=True         # Draw line with '+' from center to COM of internal nodes
        margin_ratio=0.05    # Extra margin around the root box
        fig, ax = plt.subplots(figsize=figsize)
        # Plot node bounds
        Gadget._drawNode(ax, self.root, showCom)
        # collect Bodies from leaves
        A = self.getBodies()
        if A != []:
            xs, ys = zip(*[(A[i].x,A[i].y) for i in range(len(A))])
            ax.plot(xs, ys, linestyle='none', marker='o', markersize=3)
        # Set view limits from root box (with margin)
        x0, y0, x1, y1, mx, my = self.root.box.asTuple()
        dx, dy = x1-x0, y1-y0
        pad_x = margin_ratio * max(dx, 1e-12)
        pad_y = margin_ratio * max(dy, 1e-12)
        ax.set_xlim(x0-pad_x, x1+pad_x)
        ax.set_ylim(y0-pad_y, y1+pad_y)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

def append(A,k):
    B = [None for _ in range(len(A)+1)]
    for i in range(len(A)): B[i]=A[i]
    B[len(A)]=k
    return B
# #Testing
# # Test 1: 4 equal-mass Bodies on the edges of a square
# P = [None]*4
# P[0] = Body(1,50,50)
# P[1] = Body(1,-50,50)
# P[2] = Body(1,-50,-50)
# P[3] = Body(1,50,-50)

    
# g = Gadget(Box.getBox(P))
# for p in P: g.add(p)
# print(g)
# g.plot()

# print("Now remove P[0]")
# g.remove(P[0])
# print(g) 
# g.plot()

# print("Now remove all but the last body")
# g.remove(P[1]); g.remove(P[2])
# print(g) 
# g.plot()

# print("Test 2: 4 unequal-mass Bodies on the edges of a square")
# P = [None]*4
# P[0] = Body(1,100,100,0,0)
# P[1] = Body(2,-100,100,0,0)
# P[2] = Body(3,-100,-100,0,0)
# P[3] = Body(4,100,-100,0,0)
    
# g = Gadget(Box.getBox(P))
# for p in P: g.add(p)
# print(g)
# g.plot()

# print("Test 3: 20 Bodies spread on a circle")
# P = [None]*20
# for i in range(20):
#     x  = 50.0 * math.cos(i * 0.3)
#     y  = 50.0 * math.sin(i * 0.3)
#     vx =  5.0 * math.cos(i * 0.7)
#     vy =  5.0 * math.sin(i * 0.7)
#     m  = 1.0 + (i % 5)            # masses: 1..5
#     P[i] = Body(m, x, y, vx, vy)

# g = Gadget(Box(-50,-50,50,50))
# for p in P: g.add(p)
# print(g); g.plot()

# for i in range(10): 
#     print(i)
#     print(P[i])
#     g.remove(P[i])
#     if P[i] in g.getBodies():
#         print("False success")
    

# print(g); g.plot()
####################################################################
# Question 3: Using the Gadget to speed up simulations
# To solve: get bodies from a Gadget wrt reference body 
####################################################################

class FastSimulation(Simulation):
    # Runs the simulation and produces an array of arrays of Bodies.
    # The t-th entry in the list is the position of Bodies after the t-th timestep
    # has been simulated.
    # Simulation uses a Gadget in order to approximate for each body the list 
    # of other bodies gravitationally affecting its trajectory.
    def run(self,test=None): 
        pss = [None]*(self.timesteps+1)
        pss[0] = self.bodies
        for t in range(self.timesteps):
            # calculate the current Gadget
            g = Gadget.fromBodies(pss[t])
            # for every Body in the current timestep add its next position in next timestep
            # but using the gadget g
            A = pss[t][:]
            for i in range(len(A)):
                new_ps = FastSimulation.getBodies(g,A[i])
                A[i] = A[i].next(new_ps,self.dt)
                if test is not None: test[i] = new_ps
            pss[t+1] = A
        return pss

    # Barnes-Hut criterion deciding whether node n should be opened when calculating
    # the gravitational forces exerted on body p.
    # By defualt, if p is in n then we open.
    @staticmethod
    def BarnesHut(n,p):
        theta = 0.7
        if n.box.isIn(p): return True
        sqDist = (n.COM.x-p.x)**2+(n.COM.y-p.y)**2
        return n.box.maxSide**2/sqDist >= theta**2
        
    # Get bodies from gadget g in an optimal way, using criterion shouldOpen and body p to 
    # decide whether nodes in the gadget should be "opened" (i.e. their subnodes examined)
    # or not (i.e. the whole subtree approximated by its COM).
    @staticmethod
    def getBodies(g,p,shouldOpen=BarnesHut): # 20%
        new_ps = []
        # Depth first
        stack = Stack()
        stack.push(g.root)
        
        while not stack.isEmpty():
            node = stack.pop()

            if node.isLeaf():
                if node.p:
                    new_ps = append(new_ps, node.p)

            elif shouldOpen(node, p):
                for child in node.children:
                    stack.push(child)

            else:
                new_ps = append(new_ps, node.COM)

        return new_ps
        
# Append helper function
def append(l, new_value):
    new_l = [None] * (len(l)+1)
    for i, v in enumerate(l):
        new_l[i] = v
    new_l[-1] = new_value
    return new_l

####################################################################
# Some testing code
# DO NOT INCLUDE TESTING CODE IN YOUR SUBMISSION
####################################################################

# First, some cool simulations!

# Include this line for rendering animation in Jupyter notebook 
# Note: need to install jupyter-widgets-jupyterlab-manager
# %matplotlib ipympl
# Otherwise, save as .py and run in terminal / IDE to see animations

# a sun and a planet
P=[Body(1,0,0,0,0),Body(3.0e-6,0,1,4.44,4.44)]
sim = Simulation(P)
sim.show(-4,-4,4,4)

# 3 suns and a planet
sA = Body(10,-10,0,3.14,-5.44)
sB = Body(10,0,0,3.24,5.44)
sC = Body(10,-5,8.660,-6.28,0)
p  = Body(3.0e-6,-10,6,-3,-15)

Bodies=[sA,sB,sC,p]
sim = Simulation(Bodies)
sim.show(-30,-30,30,30) 

# 81 stationary bodies on a grid
P = [None]*81
for y in range(9):
    for x in range(9):
        P[9*y+x] = Body(1,2*x,2*y)

sim = Simulation(P, total_time=1)
sim.show(0,0,20,20)

# 42 moving bodies starting on a circle
P = [None]*42
for i in range(len(P)):
    x  = 50.0 * math.cos(i * 0.3)
    y  = 50.0 * math.sin(i * 0.3)
    vx = 50.0 * math.cos(i * 0.7)
    vy = 50.0 * math.sin(i * 0.7)
    m  = 1.0 + (i % 5)            # masses: 1..5
    P[i] = Body(m, x, y, vx, vy)

sim = Simulation(P)
sim.show(-750,-750,750,750)

# 84 moving bodies starting on two circles
P = [None]*84
for i in range(len(P)//2):
    x  = 50.0 * math.cos(i * 0.3)
    y  = 50.0 * math.sin(i * 0.3)
    vx = 50.0 * math.cos(i * 0.7)
    vy = 50.0 * math.sin(i * 0.7)
    m  = 1.0 + (i % 5)            # masses: 1..5
    P[2*i] = Body(m, x, y, 2*vx, 2*vy)
    P[2*i+1] = Body(m, 2*x, 2*y, vx, vy)

sim = Simulation(P)
sim.show(-750,-750,750,750)


# Question 1

# Tests for Simulation.closestDistance

P = [None]*81
for y in range(9):
    for x in range(9):
        P[9*y+x] = Body(1,2*x,2*y)

sim = Simulation(P)
print(sim.closestDistance())

sim = Simulation(P, total_time=0.5, dt=0.0001)
print(sim.closestDistance())

# Tests for Body.threeBodyProblem

sA = Body(10,-10,0,3.14,-5.44,)
sB = Body(10,0,0,3.24,5.44)
sC = Body(10,-5,8.660,-6.28,0)
p  = Body(3.0e-6,-10,6,-3,-15)
    
Bodies=[sA,sB,sC,p]
print(p.threeBodyProblem(sA,sB,sC,50,50,50))
print(p.threeBodyProblem(sA,sB,sC,45,45,45))


# Question 2
    
# Test 1: 4 equal-mass Bodies on the edges of a square
P = [None]*4
P[0] = Body(1,50,50)
P[1] = Body(1,-50,50)
P[2] = Body(1,-50,-50)
P[3] = Body(1,50,-50)
    
g = Gadget(Box.getBox(P))
for p in P: g.add(p)
print(g)
g.plot()

# Now remove P[0]
g.remove(P[0])
print(g) 
g.plot()

# Now remove all but the last body
g.remove(P[1]); g.remove(P[2])
print(g) 
g.plot()

# Test 2: 4 unequal-mass Bodies on the edges of a square
P = [None]*4
P[0] = Body(1,100,100,0,0)
P[1] = Body(2,-100,100,0,0)
P[2] = Body(3,-100,-100,0,0)
P[3] = Body(4,100,-100,0,0)
    
g = Gadget(Box.getBox(P))
for p in P: g.add(p)
print(g)
g.plot()

# Test 3: 20 Bodies spread on a circle
P = [None]*20
for i in range(20):
    x  = 50.0 * math.cos(i * 0.3)
    y  = 50.0 * math.sin(i * 0.3)
    vx =  5.0 * math.cos(i * 0.7)
    vy =  5.0 * math.sin(i * 0.7)
    m  = 1.0 + (i % 5)            # masses: 1..5
    P[i] = Body(m, x, y, vx, vy)

g = Gadget(Box(-50,-50,50,50))
for p in P: g.add(p)
print(g); g.plot()

for i in range(10): g.remove(P[i])
print(g); g.plot()


# Question 3

def testOnThreeCircles(n):
    P = [None]*3*n
    for i in range(n):
        x  = 50.0 * math.cos(i * 0.3)
        y  = 50.0 * math.sin(i * 0.3)
        vx = 50.0 * math.cos(i * 0.7)
        vy = 50.0 * math.sin(i * 0.7)
        m  = 1.0 + (i % 5)            # masses: 1..5
        P[3*i] = Body(m, x, y, 3*vx, 3*vy)
        P[3*i+1] = Body(m, 2*x, 2*y, 2*vx, 2*vy)
        P[3*i+2] = Body(m, 3*x, 3*y, vx, vy)
    sim = FastSimulation(P, total_time=0.01)
    new_ps = [None]*len(P)
    sim.run(test=new_ps)
    print(f"Experiment with n={n}, new_ps entries:")
    for ps in new_ps: print(len(ps),ps)

testOnThreeCircles(1)
testOnThreeCircles(3)