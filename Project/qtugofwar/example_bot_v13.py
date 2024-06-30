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
    '''
        STRATEGY:
        1. To make an informed decision, we need to know the current state of the system (qubit probabilities). For that, I compute the probabilities for states 0 and 1 each time before I do something.
        2. I create 2 sets of cards, one if the set of cards that are not so significant (usable_cards) and should be used before the final rounds so, that there is room for more significant cards in the hand near the end of the gameplay. Other, is the set of significant cards which are the cards that can only be played near the end of the gameplay.
        3. At the start of the game (round 0), we can play any card in hand according to increasing signficance of the cards as a part of set known as "usable_cards".
        4. When we are near the end of the game (round >= 99), we only play significant cards. To decide which card would benefit us the most, we calculate the state post if a card in the hand is played. This means that at first, we calculate the impact of all the cards in hand at that point.
        5. The card which benefits us the most is played first in a greedy approach. If no card would favor us the most, we play the card which would give us the highest probability of our team winning post playing that card.
        6. We also keep track of the number of cards played at a point in time as we don't want to play all the cards before we are near the end of the gameplay. As we can get at most 20 cards during a game, I don't play any card before we are near the end of the gameplay after I have played 15 cards as that would mean that we would have the higher probability of having all the significant cards near the end of the gameplay.
        7. Also, I keep track of the number of cards in my hand at any point in time. This is to ensure that we only play the least significant card possible before we are near the end. So, we would only play (other than near the end) a card when our hand is full!
        8. This approach helps us maximize the probability of winning the game by keep most significant cards till the end and only playing the cards less significant at first. The most powerful card that we have is the X gate card as that swaps the probabilities using which we can increase the chances of us winnig by many folds. In case, we don't have an X gate card in hand, we try and reduce the probabilities of both 0 and 1 together so, as to give ourselves the chance of winning probabilistically. In case, both of these cards have been played out before the end of the game, we try and use the Z gate card which would help us in negating the probability of state 1 which would help us reduce the chances of winning of team 1 when the direction of spin is positive and vice-versa.
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
                    min_prob_inc = float("-inf")
                    next_action = None
                    
                    for curr_hand in hand:
                        if curr_hand in sig_cards and curr_hand not in state_changes_map:
                            next_state = self.perform_action(self.current_state, direction, curr_hand, None, True) 
                            state_changes_map[curr_hand] = next_state[1]**2
                    for prob_action, new_prob_1 in state_changes_map.items():
                        if new_prob_1 - self.current_state[1]**2 > 0 and new_prob_1 - self.current_state[1]**2 > min_prob_inc:
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
                else:
                    prev_action_0 = prev_turn['team0_action']
                    if prev_action_0 in hand and prev_action_0 in sig_cards:
                        return prev_action_0
                    
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
                    min_prob_inc = float("-inf")
                    next_action = None
                    
                    for curr_hand in hand:
                        if curr_hand in sig_cards and curr_hand not in state_changes_map:
                            next_state = self.perform_action(self.current_state, direction, curr_hand, None, False) 
                            state_changes_map[curr_hand] = next_state[0]**2
                    for prob_action, new_prob_0 in state_changes_map.items():
                        if new_prob_0 - self.current_state[0]**2 > 0 and new_prob_0 - self.current_state[0]**2 > min_prob_inc:
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
                else:
                    prev_action_1 = prev_turn['team1_action']
                    if prev_action_1 in hand and prev_action_1 in sig_cards:
                        return prev_action_1
                    
                    return None
                
            if self.num_actions_played >= 15 or len(hand) != 5:
                return None
            
            for usable_card in usable_cards:
                if usable_card in hand:
                    self.num_actions_played += 1
                    return usable_card
                

        return None
    
    def perform_action(self, current_state, direction, action: GameAction, measurement, perform_rotation):
        '''
        Take an action on a state.
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
    