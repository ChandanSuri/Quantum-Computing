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
        usable_cards = [GameAction.REVERSE, GameAction.MEASURE, GameAction.PAULIZ, GameAction.HADAMARD]
        sig_cards = [GameAction.PAULIX, GameAction.HADAMARD, GameAction.PAULIZ]
        
        if round_number == 0:
            for usable_card in usable_cards:
                if usable_card in hand:
                    self.num_actions_played += 1
                    return usable_card
            return None
        
        if team == 0:
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team0_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team0_action'], prev_turn['team0_measurement'], False)
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team1_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team1_action'], prev_turn['team1_measurement'], True)
        else:
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team1_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team1_action'], prev_turn['team1_measurement'], True)
            self.is_direction_positive = not(self.is_direction_positive) if prev_turn['team0_action'] == GameAction.REVERSE else self.is_direction_positive
            direction = 1 if self.is_direction_positive else -1
            self.current_state = self.perform_action(self.current_state, direction, prev_turn['team0_action'], prev_turn['team0_measurement'], False)            
        
        if team == 1:
            if round_number >= rounds_threshold:
                if self.current_state[0]**2 > self.current_state[1]**2:
                    state_changes_map = dict()
                    min_prob_inc = float("inf")
                    next_action = None
                    
                    for curr_hand in hand:
                        if curr_hand in sig_cards and curr_hand not in state_changes_map:
                            next_state = self.perform_action(self.current_state, direction, curr_hand, None, True) 
                            state_changes_map[curr_hand] = next_state[1]**2
                    for prob_action, new_prob_1 in state_changes_map.items():
                        if new_prob_1 - self.current_state[1]**2 > 0 and new_prob_1 - self.current_state[1]**2 < min_prob_inc:
                            min_prob_inc = new_prob_1 - self.current_state[1]**2
                            next_action = prob_action
                    
                    if next_action is None:
                        max_prob = 0
                        for prob_action, new_prob_0 in state_changes_map.items():
                            if new_prob_0 > max_prob:
                                max_prob = new_prob_0
                                next_action = prob_action
                    
                    self.num_actions_played = self.num_actions_played + 1 if next_action is not None else self.num_actions_played
                    return next_action
                    
                return None
                
            if self.num_actions_played >= 15 or len(hand) != 5:
                return None
            
            for usable_card in usable_cards:
                if usable_card in hand:
                    self.num_actions_played += 1
                    return usable_card
        else:
            if round_number >= rounds_threshold:
                if self.current_state[1]**2 > self.current_state[0]**2:
                    state_changes_map = dict()
                    min_prob_inc = float("inf")
                    next_action = None
                    
                    for curr_hand in hand:
                        if curr_hand in sig_cards and curr_hand not in state_changes_map:
                            next_state = self.perform_action(self.current_state, direction, curr_hand, None, False) 
                            state_changes_map[curr_hand] = next_state[0]**2
                    for prob_action, new_prob_0 in state_changes_map.items():
                        if new_prob_0 - self.current_state[0]**2 > 0 and new_prob_0 - self.current_state[0]**2 < min_prob_inc:
                            min_prob_inc = new_prob_0 - self.current_state[0]**2
                            next_action = prob_action
                            
                    if next_action is None:
                        max_prob = 0
                        for prob_action, new_prob_0 in state_changes_map.items():
                            if new_prob_0 > max_prob:
                                max_prob = new_prob_0
                                next_action = prob_action
                    
                    self.num_actions_played = self.num_actions_played + 1 if next_action is not None else self.num_actions_played
                    return next_action
                    
                return None
                
            if self.num_actions_played >= 15 or len(hand) != 5:
                return None
            
            for usable_card in usable_cards:
                if usable_card in hand:
                    self.num_actions_played += 1
                    return usable_card
                

        #######################################################
        return None
    
    def perform_action(self, current_state, direction, action: GameAction, measurement, perform_rotation):
        '''
        Take an action on a state (and report the result of a measurement to a strategy if it measured).
        '''
        theta = (direction * 2 * np.pi)/100.0
        rotation_matrix = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
        new_state = current_state.copy()
        
        if action == GameAction.MEASURE:
            new_state = measurement
        elif action == GameAction.PAULIX:
            X = np.array([[0, 1], [1, 0]])
            new_state = np.dot(X, current_state)
        elif action == GameAction.PAULIZ:
            Z = np.array([[1, 0], [0, -1]])
            new_state = np.dot(Z, current_state)
        elif action == GameAction.HADAMARD:
            H = np.array([[np.sqrt(1/2), np.sqrt(1/2)], [np.sqrt(1/2), -np.sqrt(1/2)]])
            new_state = np.dot(H, current_state)
        
        if perform_rotation:
            new_state = np.dot(rotation_matrix, new_state)
            
        return new_state
    