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
        self.num_actions_played = 0
    
    def play_action(self,
                    team: int,
                    round_number: int,
                    hand: List[GameAction],
                    prev_turn: List) -> Optional[GameAction]:
        

        ##### IMPLEMENT AWESOME STRATEGY HERE ##################
        
        # 4 gates possible: GameAction.PAULIX, GameAction.PAULIZ, GameAction.HADAMARD, GameAction.REVERSE, GameAction.MEASURE
        
        rounds_threshold = 99
        if round_number == 0:
            if GameAction.PAULIZ in hand:
                self.num_actions_played += 1
                return GameAction.PAULIZ
            elif GameAction.REVERSE in hand:
                self.num_actions_played += 1
                return  GameAction.REVERSE
            elif GameAction.MEASURE in hand:
                self.num_actions_played += 1
                return GameAction.MEASURE
            elif GameAction.HADAMARD in hand:
                self.num_actions_played += 1
                return GameAction.HADAMARD
            else:
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
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team0_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team0_action'], prev_turn['team0_measurement'])            
        
        if team == 1:
            if round_number >= rounds_threshold:
                if self.current_state[0]**2 > self.current_state[1]**2:
                    self.num_actions_played += 1
                    return GameAction.PAULIX if GameAction.PAULIX in hand else None
                
            if self.num_actions_played >= 15 or len(hand) != 5:
                return None
            
            for hand_idx in range(len(hand) - 1, -1, -1):
                if hand[hand_idx] in [GameAction.PAULIZ, GameAction.HADAMARD, GameAction.REVERSE, GameAction.MEASURE]:
                    self.num_actions_played += 1
                    return hand[hand_idx]
        else:
            if round_number >= rounds_threshold:
                if self.current_state[0]**2 < self.current_state[1]**2:
                    self.num_actions_played += 1
                    return GameAction.PAULIX if GameAction.PAULIX in hand else None
                
            if self.num_actions_played >= 15 or len(hand) != 5:
                return None
            
            for hand_idx in range(len(hand) - 1, -1, -1):
                if hand[hand_idx] in [GameAction.PAULIZ, GameAction.HADAMARD, GameAction.REVERSE, GameAction.MEASURE]:
                    self.num_actions_played += 1
                    return hand[hand_idx]
                

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
    