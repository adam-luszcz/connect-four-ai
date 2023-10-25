import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


"""
The program is a fuzzy logic controller for a music player.
The controlled output is music player volume.
The system is desined for users listening to music while falling asleep.
It takes into account 3 inputs: user heart beat, surrounding noise and music beat rate.
The output is the volume of the music player.


How to set up and run the program:
---
Please install skfuzzy with `pip3 install skfuzzy`
Please install skfuzzy with `pip3 install -U scikit-fuzzy`
and matplotlib with `pip3 install matplotlib`
and run the game with: `app.py`


Authors: Adam Łuszcz, Anna Rogala
"""


def music_volume_controller():
    x_heart_beat = np.arange(0, 11, 1)
    x_surrounding_noise = np.arange(0, 11, 1)
    x_music_beat_rate = np.arange(0, 11, 1)
    x_music_volume = np.arange(0, 51, 1)

    heart_beat_lo = fuzz.trimf(x_heart_beat, [0, 0, 5])
    heart_beat_md = fuzz.trimf(x_heart_beat, [0, 5, 10])
    heart_beat_hi = fuzz.trimf(x_heart_beat, [5, 10, 10])

    surrounding_noise_lo = fuzz.trimf(x_surrounding_noise, [0, 0, 5])
    surrounding_noise_md = fuzz.trimf(x_surrounding_noise, [0, 5, 10])
    surrounding_noise_hi = fuzz.trimf(x_surrounding_noise, [5, 10, 10])

    music_beat_rate_lo = fuzz.trimf(x_music_beat_rate, [0, 0, 5])
    music_beat_rate_md = fuzz.trimf(x_music_beat_rate, [0, 5, 10])
    music_beat_rate_hi = fuzz.trimf(x_music_beat_rate, [5, 10, 10])

    music_volume_lo = fuzz.trimf(x_music_volume, [0, 0, 25])
    music_volume_md = fuzz.trimf(x_music_volume, [0, 25, 50])
    music_volume_hi = fuzz.trimf(x_music_volume, [25, 50, 50])

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

    ax0.plot(x_heart_beat, heart_beat_lo, 'b', linewidth=1.5, label='Low')
    ax0.plot(x_heart_beat, heart_beat_md, 'g', linewidth=1.5, label='Medium')
    ax0.plot(x_heart_beat, heart_beat_hi, 'r', linewidth=1.5, label='High')
    ax0.set_ylim(0, 1)
    ax0.set_title('Heart beat')
    ax0.legend()

    ax1.plot(x_surrounding_noise, surrounding_noise_lo, 'b', linewidth=1.5, label='Low')
    ax1.plot(x_surrounding_noise, surrounding_noise_md, 'g', linewidth=1.5, label='Medium')
    ax1.plot(x_surrounding_noise, surrounding_noise_hi, 'r', linewidth=1.5, label='High')
    ax1.set_ylim(0, 1)
    ax1.set_title('Surrounding noise')
    ax1.legend()

    ax2.plot(x_music_beat_rate, music_beat_rate_lo, 'b', linewidth=1.5, label='Low')
    ax2.plot(x_music_beat_rate, music_beat_rate_md, 'g', linewidth=1.5, label='Medium')
    ax2.plot(x_music_beat_rate, music_beat_rate_hi, 'r', linewidth=1.5, label='High')
    ax2.set_ylim(0,1)
    ax2.set_title('Music beat rate')
    ax2.legend()

    ax3.plot(x_music_volume, music_volume_lo, 'b', linewidth=1.5, label='Low')
    ax3.plot(x_music_volume, music_volume_md, 'g', linewidth=1.5, label='Medium')
    ax3.plot(x_music_volume, music_volume_hi, 'r', linewidth=1.5, label='High')
    ax3.set_ylim(0,1)
    ax3.set_title('Music volume')
    ax3.legend()

    for ax in (ax0, ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()

    heart_beat_level_lo = fuzz.interp_membership(x_heart_beat, heart_beat_lo, 9.9)
    heart_beat_level_md = fuzz.interp_membership(x_heart_beat, heart_beat_md, 9.9)
    heart_beat_level_hi = fuzz.interp_membership(x_heart_beat, heart_beat_hi, 9.9)

    surrounding_noise_level_lo = fuzz.interp_membership(x_surrounding_noise, surrounding_noise_lo, 6.5)
    surrounding_noise_level_md = fuzz.interp_membership(x_surrounding_noise, surrounding_noise_md, 6.5)
    surrounding_noise_level_hi = fuzz.interp_membership(x_surrounding_noise, surrounding_noise_hi, 6.5)

    music_beat_rate_level_lo = fuzz.interp_membership(x_music_beat_rate, music_beat_rate_lo, 4.2)
    music_beat_rate_level_md = fuzz.interp_membership(x_music_beat_rate, music_beat_rate_md, 4.2)
    music_beat_rate_level_hi = fuzz.interp_membership(x_music_beat_rate, music_beat_rate_hi, 4.2)
    music_volume_activation_lo = np.fmin(heart_beat_level_lo, music_volume_lo)

    min_of_heart_beat_and_surrounding_noise_md = np.fmin(heart_beat_level_md, surrounding_noise_level_md)
    active_rule1 = np.fmax(min_of_heart_beat_and_surrounding_noise_md, heart_beat_level_md)
    music_volume_activation_md = np.fmin(active_rule1, music_volume_md)

    min_of_heart_beat_and_surrounding_noise_hi = np.fmin(heart_beat_level_hi, surrounding_noise_level_hi)
    active_rule2 = np.fmax(min_of_heart_beat_and_surrounding_noise_hi, heart_beat_level_hi)
    music_volume_activation_hi = np.fmin(active_rule2, music_volume_hi)

    music_volume0 = np.zeros_like(x_music_volume)

    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_music_volume, music_volume0, music_volume_activation_lo, facecolor='b', alpha=0.7)
    ax0.plot(x_music_volume, music_volume_lo, 'b', linewidth=0.5, linestyle='--', )
    ax0.fill_between(x_music_volume, music_volume0, music_volume_activation_md, facecolor='g', alpha=0.7)
    ax0.plot(x_music_volume, music_volume_md, 'g', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_music_volume, music_volume0, music_volume_activation_hi, facecolor='r', alpha=0.7)
    ax0.plot(x_music_volume, music_volume_hi, 'r', linewidth=0.5, linestyle='--')
    ax0.set_title('Output membership activity')

    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()

    aggregated = np.fmax(music_volume_activation_lo, np.fmax(music_volume_activation_md, music_volume_activation_hi))

    music_volume = fuzz.defuzz(x_music_volume, aggregated, 'centroid')
    music_volume_activation = fuzz.interp_membership(x_music_volume, aggregated, music_volume)  # for plot
    print(music_volume)

    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_music_volume, music_volume_lo, 'b', linewidth=0.5, linestyle='--', )
    ax0.plot(x_music_volume, music_volume_md, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(x_music_volume, music_volume_hi, 'r', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_music_volume, music_volume0, aggregated, facecolor='Orange', alpha=0.7)
    ax0.plot([music_volume, music_volume], [0, music_volume_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Aggregated membership and result (line)')

    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    music_volume_controller()
