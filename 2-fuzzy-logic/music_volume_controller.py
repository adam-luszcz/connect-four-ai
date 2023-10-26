"""
The program is a fuzzy logic controller for a music player volume.
It bases on 3 inputs:
- user heart beat as beat per minute,
- surrounding noise as decibels
- music beat rate as beat per minute.
The output is music player volume as percentage of full music volume.

The system is desined for users listening to music while falling asleep.


How to set up
---
Please install skfuzzy with `pip3 install -U scikit-fuzzy`
and matplotlib with `pip3 install matplotlib`


How to run
---
Please set the inputs in the last lines of the program:
music_volume_sim.input['heart_beat'] = <give_your_heart_beat>
music_volume_sim.input['surrounding_noise'] = <give_surrounding_noise>
music_volume_sim.input['music_beat_rate'] = <give_music_beat_rate>

save and run the program with: `python3 app.py`


Authors: Adam Łuszcz, Anna Rogala
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

heart_beat = ctrl.Antecedent(np.arange(40, 101, 1), 'heart_beat')
surrounding_noise = ctrl.Antecedent(np.arange(20, 141, 1), 'surrounding_noise')
music_beat_rate = ctrl.Antecedent(np.arange(20, 201, 10), 'music_beat_rate')
music_volume = ctrl.Consequent(np.arange(0, 51, 1), 'music_volume')


heart_beat['low'] = fuzz.trimf(heart_beat.universe, [40, 40, 60])
heart_beat['medium'] = fuzz.trimf(heart_beat.universe, [40, 60, 100])
heart_beat['high'] = fuzz.trimf(heart_beat.universe, [60, 100, 100])

surrounding_noise['low'] = fuzz.trimf(surrounding_noise.universe, [20, 20, 80])
surrounding_noise['medium'] = fuzz.trimf(surrounding_noise.universe, [20, 80, 140])
surrounding_noise['high'] = fuzz.trimf(surrounding_noise.universe, [80, 140, 140])

music_beat_rate['low'] = fuzz.trimf(music_beat_rate.universe, [20, 20, 80])
music_beat_rate['medium'] = fuzz.trimf(music_beat_rate.universe, [20, 100, 180])
music_beat_rate['high'] = fuzz.trimf(music_beat_rate.universe, [80, 200, 200])

music_volume.automf(3, names=['low', 'medium', 'high'])


heart_beat.view()
surrounding_noise.view()
music_beat_rate.view()
music_volume.view()


rule1 = ctrl.Rule(heart_beat['low'], music_volume['low'])
rule2 = ctrl.Rule(heart_beat['medium'] & (surrounding_noise['low'] & (music_beat_rate['low'] | music_beat_rate['medium'])), music_volume['low'])
rule3 = ctrl.Rule((heart_beat['high'] & surrounding_noise['low']) & (music_beat_rate['low'] | music_beat_rate['medium']), music_volume['low'])

rule4 = ctrl.Rule(heart_beat['medium'] & surrounding_noise['low'] & music_beat_rate['high'], music_volume['medium'])
rule5 = ctrl.Rule(heart_beat['medium'] & (surrounding_noise['medium'] | surrounding_noise['high']), music_volume['medium'])
rule6 = ctrl.Rule(heart_beat['high'] & surrounding_noise['low'] & music_beat_rate['high'], music_volume['medium'])
rule7 = ctrl.Rule(heart_beat['high'] & surrounding_noise['medium'], music_volume['medium'])

rule8 = ctrl.Rule(heart_beat['high'] & surrounding_noise['high'], music_volume['high'])

music_volume_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])

music_volume_sim = ctrl.ControlSystemSimulation(music_volume_ctrl)

music_volume_sim.input['heart_beat'] = 90
music_volume_sim.input['surrounding_noise'] = 106
music_volume_sim.input['music_beat_rate'] = 60

music_volume_sim.compute()

print(music_volume_sim.output['music_volume'])
music_volume.view(sim=music_volume_sim)

plt.show()
