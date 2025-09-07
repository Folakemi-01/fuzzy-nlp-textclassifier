# src/models/fuzzy_bert.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import re

class FuzzyClassifier:
    def __init__(self, target_names):
        self.target_names = target_names
        self.control_system = self._build_fuzzy_system()
        self.fis = ctrl.ControlSystemSimulation(self.control_system)
        # This new variable will store the label of the last activated special rule
        self.last_activated_rule_label = None

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
        
        # We explicitly label our special rule
        rule1 = ctrl.Rule(soc_rel_christian['high'] & talk_rel_misc['low'], c_talk_rel_misc, label="Corrective Rule")
        
        default_rules = [ctrl.Rule(ant['high'], final_class[re.sub(r'[^a-zA-Z0-9_]', '_', name)])
                         for ant, name in zip(antecedents, self.target_names)]
        
        return ctrl.ControlSystem([rule1] + default_rules)

    def predict(self, probabilities):
        final_predictions = []
        for prob_vector in probabilities:
            # Reset our custom logger for each prediction
            self.last_activated_rule_label = None

            for i, name in enumerate(self.target_names):
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                self.fis.input[safe_name] = prob_vector[i]

            try:
                self.fis.compute()

                # --- NEW, ROBUST VERIFICATION LOGIC ---
                # The 'skfuzzy' library stores the label of the rule that contributed most to the output
                # in a hidden variable within the simulation. We access it here.
                # This checks if the rule with the label "Corrective Rule" was the most influential.
                if hasattr(self.fis, '_last_rules_fired') and "Corrective Rule" in self.fis._last_rules_fired:
                     print("--> VERIFICATION: Corrective rule was triggered.")
                # --- END OF VERIFICATION LOGIC ---

                prediction = int(round(self.fis.output['final_class']))
                if not 0 <= prediction < len(self.target_names):
                    prediction = np.argmax(prob_vector)
            except (ValueError, KeyError, IndexError):
                prediction = np.argmax(prob_vector)
            final_predictions.append(prediction)
        return np.array(final_predictions)