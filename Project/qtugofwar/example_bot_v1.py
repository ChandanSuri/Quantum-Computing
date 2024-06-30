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

    def play_action(self,
                    team: int,
                    round_number: int,
                    hand: List[GameAction],
                    prev_turn: List) -> Optional[GameAction]:


        ##### IMPLEMENT AWESOME STRATEGY HERE ##################
        p = 0.3
        
        if team == 0:
            if round_number == 0:
                if GameAction.HADAMARD in hand:
                    return GameAction.HADAMARD
                elif GameAction.PAULIZ in hand:
                    return GameAction.PAULIZ
                elif GameAction.REVERSE in hand:
                    return GameAction.REVERSE
            elif prev_turn['team1_action'] == GameAction.MEASURE or prev_turn['team0_action'] == GameAction.MEASURE:
                prev_measurement = prev_turn['team1_measurement'] if prev_turn['team1_action'] == GameAction.MEASURE else prev_turn['team0_measurement']
                if prev_measurement[0]**2 < prev_measurement[1]**2:
                    if GameAction.PAULIX in hand:
                        return GameAction.PAULIX
                    elif GameAction.PAULIZ in hand:
                        return GameAction.PAULIZ 
                    elif GameAction.HADAMARD in hand:
                        return GameAction.HADAMARD
            elif prev_turn['team1_action'] != GameAction.MEASURE and prev_turn['team1_action'] in hand:
                return prev_turn['team1_action']
            elif prev_turn['team1_action'] not in hand:
                return GameAction.REVERSE if GameAction.REVERSE in hand else None
        else:
            # In this case, we need to increase 1's probability, the opposite of what we are doing above.
            if prev_turn['team1_action'] == GameAction.MEASURE or prev_turn['team0_action'] == GameAction.MEASURE:
                prev_measurement = prev_turn['team1_measurement'] if prev_turn['team1_action'] == GameAction.MEASURE else prev_turn['team0_measurement']
                if prev_measurement[0]**2 > prev_measurement[1]**2:
                    if GameAction.PAULIX in hand:
                        return GameAction.PAULIX
                    elif GameAction.PAULIZ in hand:
                        return GameAction.PAULIZ 
                    elif GameAction.HADAMARD in hand:
                        return GameAction.HADAMARD
            elif len(hand) > 0 and np.random.random() < p:
                action = random.choice(hand)
                return action
                

        #######################################################
        return None
