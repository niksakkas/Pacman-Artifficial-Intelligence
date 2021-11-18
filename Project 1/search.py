# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from util import PriorityQueue
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    from util import Stack

    stack = Stack()
    visited_points = []
    point_path = []
    current_point_path = []

    # Add starting point to stack
    stack.push((problem.getStartState(), current_point_path))

    while(True):
        # If the stack is empty there is no solution so return []
        if stack.isEmpty():
            return []
        # print(path)
        # Get current point info (position,direction)
        point_location, point_path = stack.pop()
        visited_points.append(point_location)

        # If the point is the goal, return the path
        if problem.isGoalState(point_location):
            return (point_path)

        # Get the successors of the current point..
        successors = problem.getSuccessors(point_location)

        # ..and add each one in the stack
        for x in successors:
            # print(x)
            # If the point hasnt already been visited, add it to the stack
            if x[0] not in visited_points:
                current_point_path = point_path + [x[1]]
                stack.push((x[0], current_point_path))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    from util import Queue

    queue = Queue()
    visited_points = []
    point_path = []
    current_point_path = []

    # Add starting point to queue
    queue.push((problem.getStartState(), current_point_path))

    while(True):
        # If the stack is empty there is no solution so return []
        if queue.isEmpty():
            return []
        # print(path)
        # Get current point info (position,direction)
        point_location, point_path = queue.pop()
        visited_points.append(point_location)

        # If the point is the goal, return the path
        if problem.isGoalState(point_location):
            return (point_path)

        # Get the successors of the current point..
        successors = problem.getSuccessors(point_location)

        # ..and add each one in the queue
        for x in successors:
            # If the point hasnt already been visited, or it is not in the queue already add it to the queue
            if x[0] not in visited_points and x[0] not in (point[0] for point in queue.list):
                current_point_path = point_path + [x[1]]
                queue.push((x[0], current_point_path))
                # print(queue.list)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue

    priority_queue = PriorityQueue()
    visited_points = []
    point_path = []
    current_point_path = []

    # Add starting point to stack
    priority_queue.push((problem.getStartState(), []), 0)

    while(True):
        # If the priority queue is empty there is no solution so return []
        if priority_queue.isEmpty():
            return []

        # Get current point info (position,direction)
        point_location, point_path = priority_queue.pop()
        visited_points.append(point_location)

        # If the point is the goal, return the path
        if problem.isGoalState(point_location):
            return (point_path)

        # Get the successors of the current point..
        successors = problem.getSuccessors(point_location)

        # ..and add each one in the queue
        for x in successors:
            # Ignore if it's in visited
            if x[0] not in visited_points:
                # If it's not in queue just push it in (with the cost)
                if x[0] not in (point[2][0] for point in priority_queue.heap):

                    newPath = point_path + [x[1]]
                    priority_queue.push(
                        (x[0], newPath), problem.getCostOfActions(newPath))

            # If it is in the queue check if it's cheaper than the old one..
                elif x[0] in (point[2][0] for point in priority_queue.heap):
                    for point in priority_queue.heap:
                        if point[2][0] == x[0]:
                            old_cost = problem.getCostOfActions(point[2][1])
                            break

                    cost = problem.getCostOfActions(point_path + [x[1]])

                    # ..if it is cheaper update, otherwise ignore
                    if old_cost > cost:
                        current_point_path = point_path + [x[1]]
                        priority_queue.update((x[0], current_point_path), cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# astar helpers


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    priority_queue = PriorityQueue()
    visited_points = []

    priority_queue.push((problem.getStartState(), [], 0), 0)

    while(True):
        # If the priority queue is empty there is no solution so return []
        if priority_queue.isEmpty():
            return []
        point_location, point_path, prev_cost = priority_queue.pop()

        # If the point is the goal, return the path
        if problem.isGoalState(point_location):
            return point_path

        # Ignore if it's in visited
        if point_location not in visited_points:
            visited_points.append(point_location)

            # Get the successors of the current point..
            successors = problem.getSuccessors(point_location)

            for next_point, action, cost in successors:
                new_path = point_path + [action]
                g = prev_cost + cost
                f = g + heuristic(next_point, problem)
                priority_queue.push((next_point, new_path, g), f)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
