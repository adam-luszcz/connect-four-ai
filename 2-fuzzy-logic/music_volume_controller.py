"""
The program is a fuzzy logic controller for a music player volume.
It bases on 3 inputs:
- user heart beat as beat per minut,
- surrounding noise as decibels
- music beat rate as beat per minute.
The output is music player volume as percentage of full music volume.

The system is desined for users listening to music while falling asleep.


How to set up and run the program:
---
Please install skfuzzy with `pip3 install -U scikit-fuzzy`
and matplotlib with `pip3 install matplotlib`
and run the program with: `app.py`

the prompt will be shown to enter the values for:
- user heart beat as beat per minut (scale: 40-100),
- surrounding noise as decibels (scale: 20-140),
- music beat rate as beat per minute (scale: 20-200).

Authors: Adam Łuszcz, Anna Rogala
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

x_heart_beat = ctrl.Antecedent(np.arange(40, 101, 1), 'heart_beat')
x_surrounding_noise = ctrl.Antecedent(np.arange(20, 141, 1), 'surrounding_noise')
x_music_beat_rate = ctrl.Antecedent(np.arange(20, 201, 10), 'music_beat_rate')

x_music_volume = ctrl.Consequent(np.arange(0, 51, 1), 'music_volume')

x_heart_beat['low'] = fuzz.trimf(x_heart_beat.universe, [40, 40, 60])
x_heart_beat['medium'] = fuzz.trimf(x_heart_beat.universe, [40, 60, 100])
x_heart_beat['high'] = fuzz.trimf(x_heart_beat.universe, [60, 100, 100])
x_surrounding_noise['low'] = fuzz.trimf(x_surrounding_noise.universe, [20, 20, 80])
x_surrounding_noise['medium'] = fuzz.trimf(x_surrounding_noise.universe, [20, 80, 140])
x_surrounding_noise['high'] = fuzz.trimf(x_surrounding_noise.universe, [80, 140, 140])

x_music_beat_rate['low'] = fuzz.trimf(x_music_beat_rate.universe, [20, 20, 80])
x_music_beat_rate['medium'] = fuzz.trimf(x_music_beat_rate.universe, [20, 100, 180])
x_music_beat_rate['high'] = fuzz.trimf(x_music_beat_rate.universe, [80, 200, 200])

x_music_volume.automf(3, names=['low', 'medium', 'high'])

x_heart_beat.view()
x_surrounding_noise.view()
x_music_beat_rate.view()
x_music_volume.view()

rule1 = ctrl.Rule(x_heart_beat['low'], x_music_volume['low'])
rule2 = ctrl.Rule(x_heart_beat['medium'] & (x_surrounding_noise['low'] & (x_music_beat_rate['low'] | x_music_beat_rate['medium'])), x_music_volume['low'])
rule3 = ctrl.Rule((x_heart_beat['high'] & x_surrounding_noise['low']) & (x_music_beat_rate['low'] | x_music_beat_rate['medium']), x_music_volume['low'])
rule4 = ctrl.Rule(x_heart_beat['medium'] & x_surrounding_noise['low'] & x_music_beat_rate['high'], x_music_volume['medium'])
rule5 = ctrl.Rule(x_heart_beat['medium'] & (x_surrounding_noise['medium'] | x_surrounding_noise['high']), x_music_volume['medium'])
rule6 = ctrl.Rule(x_heart_beat['high'] & x_surrounding_noise['low'] & x_music_beat_rate['high'], x_music_volume['medium'])
rule7 = ctrl.Rule(x_heart_beat['high'] & x_surrounding_noise['medium'], x_music_volume['medium'])

rule8 = ctrl.Rule(x_heart_beat['high'] & x_surrounding_noise['high'], x_music_volume['high'])

x_music_volume_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])

x_music_volume_sim = ctrl.ControlSystemSimulation(x_music_volume_ctrl)

x_music_volume_sim.input['heart_beat'] = 90
x_music_volume_sim.input['surrounding_noise'] = 106
x_music_volume_sim.input['music_beat_rate'] = 60

x_music_volume_sim.compute()

print(x_music_volume_sim.output['music_volume'])
x_music_volume.view(sim=x_music_volume_sim)

plt.show()
