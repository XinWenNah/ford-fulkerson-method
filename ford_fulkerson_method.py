import math
"""
Name: Nah Xin Wen
"""
class Queue():
    """ Abstract class for a generic Queue. """

    def __init__(self) :
        """
        Function description: Constructor of Queue class

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Aux space complexity : O(1)
        Space Complexity:   O(1)
        """
        self.length = 0

   
    def append(self, item) :

        """
        Function description:   Adds an element to the rear of the queue.

        Input:  Item

        Ouput:  None

        Time Complexity:    O(1)
        Auxiliary Space Complexity: O(1)
        Space Complexity:   O(1)(item)
        """
        pass

    
    def serve(self):
       
        """
        Function description: Deletes and returns the element at the queue's front

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Space Complexity:   O(1)
        """
        
        pass

    def __len__(self):
        """
        Function description: Returns the number of elements in the queue.

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Space Complexity:   O(1)
        """
        
        return self.length

    def is_empty(self):
        """
        Function description: Returns True if the queue is empty

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Space Complexity:   O(1)
        """
        
        return len(self) == 0

    
    def is_full(self):
        """
        Function description: Returns True if the stack is full and no element can be pushed

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Space Complexity:   O(1)
        """
        pass


class CircularQueue(Queue):
    #From FIT1008
    """ Circular implementation of a queue with arrays.

    Attributes:
         length (int): number of elements in the stack (inherited)
         front (int): index of the element at the front of the queue
         rear (int): index of the first empty space at the back of the queue
         array (ArrayR[T]): array storing the elements of the queue

    ArrayR cannot create empty arrays. So MIN_CAPACITY used to avoid this.
    """
    MIN_CAPACITY = 1

    def __init__(self, max_capacity):
        """
        Function description: Constructor of CircularQueue class

        Input:  max_capacity(int)

        Ouput:  None

        Time Complexity:    O(n), where n is the max_capacity
        -Analysis: Creating the self.array needs max_capacity times


        Space Auxiliary Complexity: O(n), where n is the max_capacity
        -Analysis: Creating the self.array consumes max_capacity spaces

        Space Complexity:  O(1)(Input) + O(n), where n is the max_capacity
        """ 
        Queue.__init__(self)
        self.front = 0
        self.rear = 0
        self.array = [None]*(max(self.MIN_CAPACITY,max_capacity))

    def append(self, item):
        """
        Function description: 
        Adds an element to the rear of the queue 
        if the queue is not full
        else raise exception

        Input:  item

        Ouput:  None

        Time Complexity:    O(1)
        Space Auxiliary Complexity: O(1)
        Space Complexity:   O(1)
        """
        if self.is_full():
            raise Exception("Queue is full")

        self.array[self.rear] = item
        self.length += 1
        self.rear = (self.rear + 1) % len(self.array)

    def serve(self):
        """
        Function description: 
        Deletes and returns the element at the queue's front 
        if the queue is not empty else raise exception

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Space Auxiliary Complexity: O(1)
        Space Complexity:   O(1)
        """
        if self.is_empty():
            raise Exception("Queue is empty")

        self.length -= 1
        item = self.array[self.front]
        self.front = (self.front+1) % len(self.array)
        return item

    def is_full(self):
        """
        Function description: 
        Return True if the queue is full and no element can be appended.

        Input:  None

        Ouput:  None

        Time Complexity:    O(1)
        Space Auxiliary Complexity: O(1)
        Space Complexity:   O(1)
        """
        return len(self) == len(self.array)




class ResidualNetwork:
    """
    Class for the residual network graph
    Only created while running ford-fulkerson method
    """
    def __init__(self,  num_of_vertices):
        """
        Function description: 
        Constructor of ResidualNetwork class

        Input:  num_of_vertices(int)

        Ouput:  None

        Time Complexity:    O(n), where n is the num_of_vertices
        -Analysis: Creating the self.residual_vertices takes num_of_vertices times

        Auxiliary space Complexity: O(n), where n is the num_of_vertices
        -Analysis: Creating the self.residual_vertices consumes num_of_vertices spaces

        Space Complexity:  O(1)(Input) + O(n), where n is the num_of_vertices
        """
        self.residual_vertices = [None]*num_of_vertices
     
        for index in range(num_of_vertices):
            self.residual_vertices[index] = Vertex(index)
    
    def add_edges(self, network_vertices):
        """
        Function description: 
        Create the edges for the residual network

        Input:  network_vertices, an array contains Vertex class objects(from NetworkFlow class)

        Ouput:  None

        Time Complexity:    O(n), where n is the number of all network_vertices' edges
        -Analysis: Looping through the all network_vertices's edges 

        Auxiliary space Complexity: O(n), where n is the number of all network_vertices' edges
        -Analysis: Storing the new created ResidualEdge class objects to the residual_vertices

        Space Complexity:  O(v)(Input) + O(n), where n is the number of all network_vertices' edges and
                                                v is the number of networkflow vertices
        """
        for ver in network_vertices:
            for edg in ver.edges:
                start = self.residual_vertices[edg.start.id]
                end = self.residual_vertices[edg.dest.id]

                new_edge = ResidualEdge(start, end, edg.capacity)
                edg.residual_edge = new_edge
                start.edges.append(new_edge)


    def brth_fst_srch(self, start, end):
        """
        Function description: 
        Breadth first search to find the shortest path from start to end,
        then return the minimum flow of that path and update the forward and backward edge.

        Approach description:
        While doing breadth first search, we also compare and store the minimum flow of that path and store
        the used edge. When we found the sink during breadth first search, we can terminate early. After the
        breadth first search, we check the discovered attribute of sink vertex before doing backtracking.
        If it is false, means we havent discovered it, we can return 0 directly. Else, we do backtracking 
        to update the forward edge and UPDATE/ADD backward edge. The reason we do not add backward edges initally is
        we do not need the backward edge until we update the forward edge. In other word, we can think backward edge only
        "existed" after updating the forward edge.
        After backtracking, we return the minimum flow of that path. 

        Input:  
        argument 1: start(int)
        argument 2: end(int)

        Ouput:  flow(int)

        Time Complexity:    O(V + E), where V is the number of vertices, E is the number of edges
        -Analysis:  In worst case scenarios, we will visit all the vertices to reach sink.

                    V is the number of vertices
                    E is the number of edges

                    1.  While-loop will loop through all vertices : O(V)
                    2.  In each loop, the for-loop will loop through that vertex's edges list(including backward)
                        So we will loop through all edges:          O(2E)
                    3.  Do the backtracking to update the forward and backward edge:    O(V)
                    
                    Total : O(V) + O(2E) + O(V) = O(2V + 2E) 
                                                = O(V + E)

        Auxiliary space Complexity: O(V + E), where V is the number of vertices, E is the number of edges
        -Analysis:  Creating the CircularQueue needs O(V) spaces.
                    Adding all the backward edges needs O(E) spaces.

        Space Complexity:   O(1)(Input) + O(V + E),  where V is the number of vertices, E is the number of edges
        """

        #In the same vertex, we cant go to other vertices
        if(start == end):
            return 0
        
        found = False                                       #Indicating found the sink or not
        source =  self.residual_vertices[start]             
        queue = CircularQueue(len(self.residual_vertices))  #O(V), where V is the number of vertices
        source.discovered = True
        queue.append(source)

        # Find the path and the minimum flow from source to sink
        while(len(queue) > 0 and not found):
            
            current = queue.serve()

            #Get the all the edges of the vertex
            for edg in current.edges:#O(e), where e is the number of edges of current vertex
                
                #Only use the valid path, means the path's flow is greater than 0
                if(edg.available_flow > 0):
                    next_vertex = edg.dest
                    
                    if(not next_vertex.discovered):
                        next_vertex.discovered = True
                        next_vertex.previous = current

                        #Store the current minimum flow of that path in the vertex
                        next_vertex.possible_min_flow = min(edg.available_flow, current.possible_min_flow)

                        #Store the edge that is used to move from current vertex to this vertex
                        next_vertex.path_to_here = edg

                        #If the next vertex is sink, we can terminate 
                        if(next_vertex.id == end):
                            found = True
                            break
                        #Else we continue searching
                        queue.append(next_vertex)
        
        #Backtracking

        #If the sink is not discovered yet, means we cant find the path from source to sink
        #So we can return 0 flow
        if(not self.residual_vertices[end].discovered):
            return 0
        
        #Else do back tracking to update the forward edge and reverse edge
        last = self.residual_vertices[end]           #In this stage, the last is sink
        minimum_flow = last.possible_min_flow #The minimum flow of the path(finalized)

        #Back track to update the forward edge and update/add the backward edge
        while last.previous is not None:
            #Update the forward edge
            last.path_to_here.available_flow -= minimum_flow

            #Add the reverse edge
            if(last.path_to_here.twin is None):
                backward = ResidualEdge(last.path_to_here.dest, last.path_to_here.start, minimum_flow)

                #Link the backward edge to the forward edge and vice versa
                last.path_to_here.twin = backward
                backward.twin = last.path_to_here

                #Add it to the destination vertex, 
                # like the flow is 5,capacity is 10, updated forward edge is (0,1,5), the reverse edge will be (1,0, 5)
                #And reversed edge will be stored inside the Vertex 1
                self.residual_vertices[last.path_to_here.dest.id].edges.append(backward)

            #Update the backward edge
            else:
                last.path_to_here.twin.available_flow += minimum_flow

            #Go to the previous vertex
            last = last.previous

        return minimum_flow  


class NetworkFlow:
    """
    The flow network graph class
    """

    def __init__(self, number_of_vertex):
        """
        Function description: 
        Constructor of NetworkFlow class

        Input:  number_of_vertex(int)

        Ouput:  None

        Time Complexity:    O(n), where n is the number_of_vertex
        -Analysis: 
                    n is the number_of_vertex
                    1. Initalize the self.vertices : O(n)
                    2. Looping through self.vertices to create vertex: O(n)

                    Total: O(n) + O(n)  = O(2n)
                                        = O(n)

        Auxiliary space Complexity: O(n), where n is the number_of_vertex
        -Analysis: Creating the self.residual_vertices needs O(n)

        Space Complexity:  O(1)(Input) + O(n), where n is the number_of_vertex
        """
      
        self.vertices = [None]*number_of_vertex 
        for index in range(number_of_vertex):   
            self.vertices[index] = Vertex(index)

    def add_edges(self, edges):
        """
        Function description: 
        Add the edges to self.vertices

        Input:  edges, an array that contains tuple that follows this format,(start, end, capacity)

        Ouput:  None

        Time Complexity:    O(n), where n is the length of edges
        -Analysis: 
                    n is the length of edges
                    1. Looping through edges array : O(n)
                    2. Create the edges and add it to vertex: O(1)

                    Total: O(n) + O(1)  = O(n)
                                        
        Auxiliary space Complexity: O(n), where n is the length of edges
        -Analysis: The number of created edges is as same as input, needs n spaces

        Space Complexity:  O(n)(Input) + O(n), where n is the length of edges
        """
        for ed in edges:
            start = self.vertices[ed[0]]
            end = self.vertices[ed[1]]
            capacity = ed[2]

            new_edge = Edge(start, end, capacity)
            start.edges.append(new_edge)

    def ford_fulkerson(self, source, sink):
        """
        Function description: 
        Doing the ford_fulkerson method to get the maximum flow of the network

        Approach description:
        We initialize the maximum flow to 0 and create the residual network graph.After that, we run a bfs first
        to check the flow is 0 or not. If it is not 0,we run the while-loop so it can continue running the bfs to 
        accumulate the maximum flow. Inside the while-loop, we will accumulate the flow then reset the vertices of
        residual network graph. The while-loop will stop once the return flow is 0. After the while-loop, we will update
        network flow graph as well. Lastly, we return the accumulated maximum flow.

        Input:  
        argument1: source(int)
        argument2: sink(int)

        Ouput:  maximum flow(int)

        Time Complexity:    O(fe), where f is the maximum flow, e is the number of edges
        -Analysis:          v is the number of vertices
                            e is the number of edges,  minimum of e = v-1, maximum of e = v^2 for simple graph
                            f is the maximum flow

                            1. Creating the residual network :                      O(v)
                            2. Adding the edges to residual network :               O(e)
                            3. Running first brth_fst_search:                       O(v+e)
                            4.Inside the while loop,
                            reset the residual network's vertex:                    O(fv)
                            5.Inside the while loop,
                            run the brth_fst_search:                                O(fv+fe)
                            6.Update the flow network:                              O(e)

                            Total: O(v)+O(e)+O(v+e)+O(fv)+O(fv+fe)+O(e) = O(2v+3e+2fv+fe)
                                                                        = O(fv+fe) #If the graph is dense, e = v^2
                                                                        = O(fv + fv^2)
                                                                        = O(fv^2)
                                                                        = O(fe)

                              
        Auxiliary space Complexity: O(v+e), where  v is the number of vertices, e is the number of edges
        -Analysis: 
                        v is the number of vertices
                        e is the number of edges

                        1. Create the residual network graph:   O(v+e)
                        2. Run the first bfs:                   O(v+e)
                        3. Inside the while-loop, running bfs:  O(v+e)

                        Total:  O(v+e)+O(v+e)+O(v+e) = O(3v+3e)
                                                     = O(v+e)

        Space Complexity:  O(1)(Input) + O(v+e), where  v is the number of vertices, e is the number of edges
        """

        maximum_flow = 0
        #Create the residual network and run bfs multiple times
        residual_network = ResidualNetwork(len(self.vertices))  #O(v), where n is the number of vertices
        residual_network.add_edges(self.vertices)               #O(e), where v is the number of edges
        augment = residual_network.brth_fst_srch(source, sink)  #O(v + e)

        #Keep running bfs until the augment is 0, means we cant find any paths from source to sink
        while (augment > 0):
            #Accumulating the augment
            maximum_flow += augment
            
            #Reset 
            for residual_vertex in residual_network.residual_vertices:
                residual_vertex.discovered = False
                residual_vertex.previous = None

            #Continue running the bfs
            augment = residual_network.brth_fst_srch(source, sink)
       
        #Update the flow of network flow graph
        #O(e), where e is the number of all edges
        for ver in self.vertices:   
            for ed in ver.edges:   
                #The new flow equals to the capacity minus forward edge's flow
                ed.flow = ed.capacity - ed.residual_edge.available_flow

        return maximum_flow
    

       
class Vertex:
    """
    Class that represent the node
    """
    def __init__(self, id):
        """
        Function description: 
        Constructor of Vertex class

        Input:  id(int)

        Ouput:  None

        Time Complexity:    O(1)
        Auxiliary space Complexity: O(1)
        Space Complexity:  O(1)
        """
        self.id = id
        self.edges = []
        self.discovered = False
        self.previous = None
        #Store the min flow of the path
        self.possible_min_flow = math.inf
        #Store the edge that is used to move from other vertex to this vertex
        self.path_to_here = None
    
class ResidualEdge:
    """
    Class for the edge of residual network
    """
    def __init__(self, sta, dest, flow):
        """
        Function description: 
        Constructor of ResidualEdge class

        Input: 
        argument1: sta(int)
        argument2: dest(int)
        argument3: flow(int)

        Ouput:  None

        Time Complexity:    O(1)
        Auxiliary space Complexity: O(1)
        Space Complexity:  O(1)
        """
        self.start = sta
        self.dest = dest
        self.available_flow = flow  
        #Store the backward edge
        self.twin = None

class Edge:
    """Class for edge of flow network"""
    def __init__(self, sta, dest, capacity):
        """
        Function description: 
        Constructor of Edge class(for flow network)

        Input: 
        argument1: sta(int)
        argument2: dest(int)
        argument3: capacity(int)

        Ouput:  None

        Time Complexity:    O(1)
        Auxiliary space Complexity: O(1)
        Space Complexity:  O(1)
        """
        self.start = sta
        self.dest = dest
        self.flow = 0
        self.capacity = capacity
        #Store the edge of residual network
        self.residual_edge = None

def assign(preferences, places):
    """
    Function description: 
    Convert the two arguments into a flownetwork graph then run ford fulkerson method to check it is feasible or not.
    If it is feasible, it will return a list of list else None.

    Approach description:  
    In order to use ford fulkerson method to solve this problem, we need to do some preprocess to convert the two lists into a network flow graph first. 
    To create the graph, we need to prepare an edges list. We treat each inner list of preferences as an individual node and each number of places list as two nodes. 
    For each place , the first node represents the demand of two interested and experienced people(which means demand of it is always 2) while the second node 
    represents the remaining demand of people, which is the original demand - 2. 
    For the first node, only people node with number 2 to that place will have an edge between the people node and this node. 
    For the second node, only people node with number 1 or number 2 to that place will have an edge between the people node and this node. People node who has edge to the
    first node still can have edge to the second node. 

    After preparing the edges between preferences and places, we need to add source and sink node, and the relevant edges.
    which are the edges between source and every people node, and edges between places node(including first and second) and sink.

    After finish preparing the edges list, we create the network flow graph then run the ford fulkerson method and store the maximum flow.
    After that, we use this maximum flow to minus all the demands of the places, and check its remaining is 0 or not. If it is 0, means 
    this problem is feasible, else we just return None since it is not feasible.

    If it is feasible, we will initalize a list and append the inner list(s), the number of inner list is as same as the people.
    After that, we will loop through the all the edges between places and people to know how the flows go so we know this person should
    be assigned to which places. This involve some calculations. Last, we return this list of list.

    Input: 
    argument1:  preferences, a list that contains list of numbers, length of inner list is same as length of places,
                numbers are between range [0,2]
    argument2: places, an array that contains numbers, which are the minimum people requirement of that place.
    
    Ouput:  a list of list or None

    Time Complexity:    O(n^3), where n is the number of people
    -Analysis:  
                n is the length of preferences
                m is the length of places, maximum m = n//2
                f is the maximum flow, if it is feasible, f = n

                Worst case, all people are interested and have experience in every activity, so all of them are number 2

                1.  For-loop to add the edges between source and people:                        O(n)
                    -In each loop, it will also add the edges between people and activities:    O(2m)
                                                                                    Total:      O(2nm)
                2.  For-loop to add the edges between activities and sink:                      O(m)
                3.  Create the flownetwork :                                                O(1+n+2m+1)
                4.  Add the edges to the flownetwork graph:                                 O(n+2nm+2m)
                5.  Run the ford fulkerson method:                                          O(f*(n+2nm+2m))
                6.  If it is feasible, for-loop to add the list to a list:                  O(m)
                7.  Loop through the edges between people and activies:                     O(2nm)

                Total:  O(2nm) + O(m) + O(1+n+2m+1) + O(n+2nm+2m) + O(f*(n+2nm+m)) 
                        + O(m)  + O(2nm)  = O(2n + 7m + 6nm + f*(n+2nm+m) + 2)
                                                = O(f*(n+2nm + m))  worst case:m = n//2
                                                = O(f*(n + 2n*n//2 + n//2))
                                                = O(f*n^2) worst case: f = n
                                                = O(n*n^2)
                                                = O(n^3)

    Auxiliary space Complexity: O(n^2), where n is the length of ppreferences
    -Analysis:  
                n is the length of preferences
                m is the length of places, maximum m is n//2

                1. Adding the edges between source and people:              O(n)
                2. Adding the edges between people and activies:            O(2nm)
                3. Adding the edges between activities and sink:            O(2m)
                4. Creating the flownetwork:                                O(1+n+2m+1)
                5. Adding the edges:                                        O(n+2nm+m)
                6. Running ford fulkerson:                                  O(1+n+2m+1 + n+2nm+m)
                7. If it is feasible, for-loop to get the arrangemnt:       O(n)

                Total:  O(n)+O(2nm)+O(2m)+O(1+n+2m+1)+O(n+2nm+m)+O(1+n+2m+1 + n+2nm+m)+O(n)
                        = O(6n + 8m + 6nm + 4)
                        = O(nm) worst case: m = n//2
                        = O(n*n//2)
                        = O(n^2)

    Space Complexity:  O(n)(input) + O(m)(input) + O(n^2), where  n is the length of preferences,
                                                                  m is the length of places, maximum m is n//2
    """


    #The edges list to create the network flow graph
    edges = []

    #The first index is source, the last index is sink
    #Adding the edges between source and people node
    for i in range(1, len(preferences)+1):
        edges.append((0, i, 1))

        #Adding the edges between people node and places nodes(including first and second)
        for j in range(len(places)):#(n//2)
            index_of_activity = 2*j+len(preferences)+1

            #People node with number 2
            if(preferences[i-1][j] == 2):
                edges.append((i, index_of_activity, 1))
                edges.append((i, index_of_activity+1, 1))

            #People node number 1
            elif(preferences[i-1][j] == 1):
                edges.append((i, index_of_activity+1, 1))
        

    #Adding the edges between sink and places nodes(including first and second)
    for j in range(len(places)):
        index_of_activity = 2*j + len(preferences)+1
        sink = 1 + len(preferences)+2*len(places)

        edges.append((index_of_activity, sink, 2))              # This is the first place node
        edges.append((index_of_activity+1, sink, places[j]-2))  # This is the second place node

    
    #Create the flow network graph and run ford fulkerson
    network = NetworkFlow(1+len(preferences)+2*len(places)+1)
    network.add_edges(edges)
    flow = network.ford_fulkerson(0, 1+len(preferences)+2*len(places))

    #Check it is feasible or not

    if(flow != len(preferences)):
        return None
    
    #If it is feasible, then initialize the return list of list
    arrangement = []
    for i in range(len(places)):
        arrangement.append([])
    
    #Looping through the edges between people nodes and places node to find how the flows go
    for i in range(1, len(preferences)+1):# For each people node
        for e in network.vertices[i].edges: 
            if(e.flow == 1):
                arrangement[(int(e.dest.id)- len(preferences)-1)//2].append(int(e.start.id)-1)
    return arrangement

