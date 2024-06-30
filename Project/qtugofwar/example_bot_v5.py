from typing import List, Optional
import numpy as np
import random
from GamePlayer import *

'''
Insert high-level explanation of your strategy here. Why did you design this strategy?
When should it work well, and when should it have trouble?
'''
class MyStrategy(GameBot):

    '''
        Initialize your bot here. The init function must take in a bot_name.
        You can use this to initialize any variables or data structures
        to keep track of things in the game
    '''
    def __init__(self,bot_name):
        self.bot_name = bot_name        #do not remove this
        self.current_state = [np.sqrt(1/2), np.sqrt(1/2)]
        self.is_direction_positive = True
    
    def play_action(self,
                    team: int,
                    round_number: int,
                    hand: List[GameAction],
                    prev_turn: List) -> Optional[GameAction]:
        

        ##### IMPLEMENT AWESOME STRATEGY HERE ##################
        
        # 4 gates possible: GameAction.PAULIX, GameAction.PAULIZ, GameAction.HADAMARD, GameAction.REVERSE, GameAction.MEASURE
        # what gate can increase the probability of zero if it's less
        # If it's less than zero, do rotation (can help change the sign)
        # If the probability for 1 is higher, use X gate
        # If the probability for 1 is higher and X gate is not available use Z.
        
        '''
        New Strategy:
        1. Just calculate the states based on previous actions.
        Team 0:
        2. If in the previous turn, measurement was done where state 1 has 1 probability, then play X (if round >=98) else play H.
        3. Else If probability of state 0 is lower, play X (but not before round 98 if we have only 1 X in hand)
        4. Else If probability of state 0 is lower, and not playing X, then play R (reverse) if R wasn't used before (or for even number of times used).
        5. Else If probability of state 0 is lower, and not playing X or R, then play Z (if state 1 had positive probability).
        6. Else If probability of state 0 is higher and near last rounds, play measure (M).
        Team 1:
        2. If in the previous turn, measurement was done where state 0 has 1 probability, then play X (if round >=98) else play H.
        3. Else If the probability of state 1 is lower, play X (but not before round 98 if we have only 1 X in hand).
        4. Else If the probability of state 1 is lower, and not playing X, then play R (reverse) if R was used odd number of times before.
        5. Else If probability of state 1 is higher and near last rounds, play measure (M).
        '''
        
        rounds_threshold = 98
        if round_number == 0:
            return None
        
        if team == 0:
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team0_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team0_action'], prev_turn['team0_measurement'])
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team1_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team1_action'], prev_turn['team1_measurement'])
        else:
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team1_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team1_action'], prev_turn['team1_measurement'])
            elf.is_direction_positive = not(self.is_direction_positive) if prev_turn['team0_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team0_action'], prev_turn['team0_measurement'])
        
        if team == 0:
            if prev_turn['team1_action'] == GameAction.MEASURE or prev_turn['team0_action'] == GameAction.MEASURE:
                prev_measurement = prev_turn['team1_measurement'] if prev_turn['team1_action'] == GameAction.MEASURE else prev_turn['team0_measurement']
                if prev_measurement[1] == 1:
                    if round_number >= rounds_threshold:
                        return GameAction.PAULIX if GameAction.PAULIX in hand else None
                    else:
                        return GameAction.HADAMARD if GameAction.HADAMARD in hand else None
            elif self.current_state[0]**2 < self.current_state[1]**2:
                if (round_number < rounds_threshold and hand.count(GameAction.PAULIX) > 1) or round_number >= rounds_threshold:
                    return GameAction.PAULIX if GameAction.PAULIX in hand else None
                elif self.is_direction_positive:
                    return GameAction.REVERSE if GameAction.REVERSE in hand else None
                elif self.current_state[1] > 0:
                    return GameAction.PAULIZ if GameAction.PAULIZ in hand else None
            elif self.current_state[0]**2 > self.current_state[1]**2 and round_number >= rounds_threshold:
                return GameAction.MEASURE if GameAction.MEASURE in hand else None
        else:
            if prev_turn['team1_action'] == GameAction.MEASURE or prev_turn['team0_action'] == GameAction.MEASURE:
                prev_measurement = prev_turn['team1_measurement'] if prev_turn['team1_action'] == GameAction.MEASURE else prev_turn['team0_measurement']
                if prev_measurement[0] == 1:
                    if round_number >= rounds_threshold:
                        return GameAction.PAULIX if GameAction.PAULIX in hand else None
                    else:
                        return GameAction.HADAMARD if GameAction.HADAMARD in hand else None
            elif self.current_state[0]**2 > self.current_state[1]**2:
                if (round_number < rounds_threshold and hand.count(GameAction.PAULIX) > 1) or round_number >= rounds_threshold:
                    return GameAction.PAULIX if GameAction.PAULIX in hand else None
                elif not(self.is_direction_positive):
                    return GameAction.REVERSE if GameAction.REVERSE in hand else None
                elif self.current_state[1] < 0 and self.current_state[0] > 0:
                    return GameAction.PAULIZ if GameAction.PAULIZ in hand else None
            elif self.current_state[1]**2 > self.current_state[0]**2 and round_number >= rounds_threshold:
                return GameAction.MEASURE if GameAction.MEASURE in hand else None
                

        #######################################################
        return None
    
    def perform_action(self, current_state, direction, action: GameAction, measurement):
        '''
        Take an action on a state (and report the result of a measurement to a strategy if it measured).
        '''
        theta = (direction * 2 * np.pi)/100.0
        rotation_matrix = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
        
        if action == GameAction.MEASURE:
            current_state = measurement
        elif action == GameAction.PAULIX:
            X = np.array([[0, 1], [1, 0]])
            current_state = np.dot(X, current_state)
        elif action == GameAction.PAULIZ:
            Z = np.array([[1, 0], [0, -1]])
            current_state = np.dot(Z, current_state)
        elif action == GameAction.HADAMARD:
            H = np.array([[np.sqrt(1/2), np.sqrt(1/2)], [np.sqrt(1/2), -np.sqrt(1/2)]])
            current_state = np.dot(H, current_state)
            
        current_state = np.dot(rotation_matrix, current_state)
            
        return current_state
    