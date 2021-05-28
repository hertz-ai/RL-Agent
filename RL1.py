# RL Agent for HertzAI
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Agent:
    def __init__(self):
        """
        Vars:
        1. self.state: {list - containing 2 lists}
            First list is material type preference.
            The second list is student material style preference. 
            Styles: Femaleness       Types: Text
                    Politeness              Audio
                    Simplicity              Video
                    Formality               Slides
                    Sentiment
                    Humour
        """
        self.state = [[], []]  # material and style preference
        self.action = []  # size is equal to number of concepts

    def selectAction(self):
        pass


class Environment:
    def materialSuggestion(self):
        """
        Summary:
        Uses the action vector and the time available to the student.
        The length of the material suggested should approximately sum up to the available time.
        The time fraction of each concept should be proportional to its value in the action vector.

        Return: 
        1. {list of dicts}: Each dict has 2 elements - {list} a material embedding
                                                     - {list} all the concepts that it deals with and the time spent on them
        """
        pass

class Estimator(nn.Module):
    def __init__(self, inputs, outputs):
        super(Estimator, self).__init__()
        
        self.hidden_layer_units = 512
        
        def linBlock(inDim, outDim):
            return nn.Sequential(
                nn.Linear(inDim, outDim),
                nn.BatchNorm2d(outDim),
                nn.LeakyReLU())
        
        self.fc1 = linBlock(inputs, self.hidden_layer_units)
        self.fc2 = linBlock(self.hidden_layer_units, self.hidden_layer_units)
        self.fc3 = linBlock(self.hidden_layer_units, self.hidden_layer_units)
        self.fc4 = linBlock(self.hidden_layer_units, outputs)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out
        
        
class Student:
    def __init__(self):
        """
        Summary:
        The items have been initialised as dictionaries since it makes it easier to understand what represents what.
        And conversion to lists/arrays is quite easy later on.

        Vars:
        1. style_preference: {dict - each element is a real number} 
            As of now there are only these many. Have set the values to None since might need sum = 1
        2. type_preference: {dict - each element is a real number} 
            question is not included as a type since I think we will keep regular questions
            meaning that we will have them regularly after every video and tests after each section/chapter/concept.
        3. state: {dict - each element is list of size 3}        
            state is also dictionary. Eg: {"Newton's Laws": [0.35, -> knowledge score
                                                             0.91, -> application score
                                                             3]    -> proficiency class
                                          }. 
            The scores are not in a dictionary since converting dict of dict to list of list is harder
            The number of classes can be easily extended, but starting with 5. 
            The proficiency class is randomly decided.
            Proficiency Classes:
            0     Noob
            1     Below Average
            2     Average
            3     Above Average
            4     Expert
        4. goal_state: {dict - each element is list of size 3}
            This has the same structure as state but tells the goal of the student. In each element of each list,
            the value must be atleast as much as the value in the corresponding place in the state at the start.
        """
        self.style_preference = {"Femaleness": None,
                                 "Politeness": None,
                                 "Simplicity": None,
                                 "Formality": None,
                                 "Sentiment": None,
                                 "Humour": None}
        self.type_preference = {"Text": None,
                                "Audio": None,
                                "Video": None,
                                "Slides": None}
        self.state = {}
        self.goal_state = {}

        self.setState()

        def setState(self):
            """
            Summary:
            This will generate a randomly initialized state and goal state for the student. 
            Leaves: nodes that do not depend on anything else.
            Assuming that the graph is a dependency graph, we start from the leaves (some - chosen randomly) and move inward.
            We assign high values to the chosen leaves and lower and lower values as we move inward.
            It is also possible that there are gaps in the student's understanding
            - meaning that later nodes have non zero values while earlier ones have 0.
            The strand moving inward from each leaf decays differently and they can merge as well.
            For goal state initialisation: we select a group of concepts to improve (randomly)
            and just take the same values as state and increase it by some non-negative amount.
            """
            pass
