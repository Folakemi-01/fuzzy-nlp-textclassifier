# src/models/fuzzy_bert.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import re

class FuzzyClassifier:
    def __init__(self, target_names):
        self.target_names = target_names
        self.fis = self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        antecedents = []
        for name in self.target_names:
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), safe_name)
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
            var['medium'] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
            var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])
            antecedents.append(var)

        final_class = ctrl.Consequent(np.arange(0, len(self.target_names), 1), 'final_class')
        for i, name in enumerate(self.target_names):
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            final_class[safe_name] = fuzz.trimf(final_class.universe, [i - 0.5, i, i + 0.5])

        soc_rel_christian = antecedents[self.target_names.index('soc.religion.christian')]
        talk_rel_misc = antecedents[self.target_names.index('talk.religion.misc')]
        c_talk_rel_misc = final_class[re.sub(r'[^a-zA-Z0-9_]', '_', 'talk.religion.misc')]
        
        rule1 = ctrl.Rule(soc_rel_christian['high'] & talk_rel_misc['low'], c_talk_rel_misc)
        default_rules = [ctrl.Rule(ant['high'], final_class[re.sub(r'[^a-zA-Z0-9_]', '_', name)])
                         for ant, name in zip(antecedents, self.target_names)]
        
        system = ctrl.ControlSystem([rule1] + default_rules)
        return ctrl.ControlSystemSimulation(system)

    def predict(self, probabilities):
        final_predictions = []
        for prob_vector in probabilities:
            for i, name in enumerate(self.target_names):
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                self.fis.input[safe_name] = prob_vector[i]
            try:
                self.fis.compute()
                prediction = int(round(self.fis.output['final_class']))
                if not 0 <= prediction < len(self.target_names):
                    prediction = np.argmax(prob_vector)
            except:
                prediction = np.argmax(prob_vector)
            final_predictions.append(prediction)
        return np.array(final_predictions)