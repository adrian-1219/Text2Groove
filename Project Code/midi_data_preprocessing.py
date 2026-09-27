# Preprocess the groove midi dataset to extract 2 bar patterns and measure groove features for each pattern. 
# Save the processed data to a h5py file for training the model.

import argparse
from pathlib import Path

import pretty_midi
import pandas as pd
import numpy as np
import h5py

WINDOW_SIZE = 2 # 2 bars per window
HOP_SIZE = 1 # 1 bar hop size
SUBDIVISION_PER_BEAT = 4
BEATS_PER_BAR = 4

STEPS_PER_BAR = BEATS_PER_BAR * SUBDIVISION_PER_BEAT
WINDOW_STEPS = WINDOW_SIZE * STEPS_PER_BAR
HOP_STEP = HOP_SIZE * STEPS_PER_BAR

# drums are ordered in a manner whereby the lower the index, the higher the priority. this is useful when consolidating multiple hits at the same time step to calculate swing features.
TO_MATRIX_MAPPING = {
    42: 0,  # closed hihat
    51: 1,  # ride cymbal
    46: 2,  # open hihat
    38: 3,  # snare
    36: 4,  # kick
    49: 5,  # crash
    50: 6,  # high tom
    48: 7,  # mid tom
    45: 8,  # low tom
}

TO_MIDI_MAPPING = {v: k for k, v in TO_MATRIX_MAPPING.items()}

# remap midi files to 9 classes (kick, snare, closed hihat, open hihat, low tom, mid tom, high tom, crash cymbal, ride cymbal)
def remap_midi_file(input_file):
    midi = pretty_midi.PrettyMIDI(input_file)
    new_midi = pretty_midi.PrettyMIDI()

    mapping = {
        36: 36,
        38: 38,
        40: 38,
        37: 38,
        48: 48,
        50: 50,
        45: 45,
        47: 48,
        43: 45,
        58: 45,
        46: 46,
        26: 46,
        42: 42,
        22: 42,
        # 44: 42,        disregard pedal hihat
        49: 49,
        55: 49,
        57: 49,
        52: 49,
        51: 51,
        59: 51,
        53: 51,
    }

    drum_inst = pretty_midi.Instrument(program=0, is_drum=True)

    for inst in midi.instruments:

        for note in inst.notes:

            if note.pitch not in mapping:
                continue

            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=mapping[note.pitch],
                start=note.start,
                end=note.end
            )

            drum_inst.notes.append(new_note)

    new_midi.instruments.append(drum_inst)

    return new_midi

# extract drum patterns of fixed length from one midi file, each pattern is represented by three matrices: hit, velocity, and offset (as described in https://proceedings.mlr.press/v97/gillick19a.html)
def extract_patterns(midi, tempo):
    seconds_per_beat = 60.0 / tempo
    step_duration = seconds_per_beat / SUBDIVISION_PER_BEAT

    total_time = midi.get_end_time()
    num_steps = int(np.ceil(total_time / step_duration))

    patterns_H = []
    patterns_O = []
    patterns_V = []

    # Ensure short MIDI files still produce one padded window
    effective_steps = max(num_steps, WINDOW_STEPS)

    for start in range(
        0,
        effective_steps - WINDOW_STEPS + 1,
        HOP_STEP,
    ):
        H = np.zeros((WINDOW_STEPS, 9), dtype=np.float32)
        O = np.zeros((WINDOW_STEPS, 9), dtype=np.float32)
        V = np.zeros((WINDOW_STEPS, 9), dtype=np.float32)

        for inst in midi.instruments:
            for note in inst.notes:
                global_t = note.start / step_duration
                global_step = int(np.round(global_t))
                local_step = global_step - start

                if not 0 <= local_step < WINDOW_STEPS:
                    continue

                # Ignore pitches that have no mapping
                if note.pitch not in TO_MATRIX_MAPPING:
                    continue

                drum_id = TO_MATRIX_MAPPING[note.pitch]

                H[local_step, drum_id] = 1.0

                offset = global_t - global_step
                velocity = note.velocity / 127.0

                # Keep the loudest note at this step/instrument
                if velocity > V[local_step, drum_id]:
                    O[local_step, drum_id] = offset
                    V[local_step, drum_id] = velocity

        patterns_H.append(H)
        patterns_O.append(O)
        patterns_V.append(V)

    return (
        patterns_H,
        patterns_O,
        patterns_V,
    )

# measure groove features of a pattern according to the methods described in https://doi.org/10.31751/1224
def measure_groove_features(pattern_H, pattern_O, pattern_V, center_offsets=True):
    # consolidate the matrices to 1D arrays 
    H_flat = np.array(32 * [0])
    O_flat = np.array(32 * [0.0])
    V_flat = np.array(32 * [0.0])
    for step in range(WINDOW_STEPS):
        for drum_id in range(9):
            if pattern_H[step, drum_id] == 1:
                H_flat[step] = 1
                O_flat[step] = pattern_O[step, drum_id]
                V_flat[step] = pattern_V[step, drum_id]
                break  # only consider the first hit in the priority order of drum voices 
    # print(H_flat)
    # print(O_flat)
    # print(V_flat)

    # compute 16th note swing density - off beat hits / number of off beat positions 
    off_beat_hits = H_flat[1::2].sum()
    off_beat_positions = len(H_flat[1::2])
    swing_density = off_beat_hits / off_beat_positions 

    # compute 8th note swing density
    off_beat_hits_8th = H_flat[2::4].sum()
    off_beat_positions_8th = len(H_flat[2::4])
    swing_density_8th = off_beat_hits_8th / off_beat_positions_8th

    even_hits = H_flat[::2].sum()
    odd_hits = H_flat[1::2].sum()

    # determine if the pattern is in 8th notes or 16th notes
    # subdivision = 8 if odd_hits / (even_hits + odd_hits) < 0.6 else 16
    
    # count the number of valid eigth note swing pairs and the number of valid sixteenth note swing pairs
    # Eighth-note swing pairs:
    # (0,2), (4,6), (8,10), ...
    eighth_starts = np.arange(0, 32, 4)
    eighth_ends = eighth_starts + 2
    valid_eighth = H_flat[eighth_starts] & H_flat[eighth_ends]
    eighth_pair_count = valid_eighth.sum()

    # Sixteenth-note swing pairs:
    # (0,1), (2,3), (4,5), ...
    sixteenth_starts = np.arange(0, 32, 2)
    sixteenth_ends = sixteenth_starts + 1
    valid_sixteenth = H_flat[sixteenth_starts] & H_flat[sixteenth_ends]
    sixteenth_pair_count = valid_sixteenth.sum()

    if sixteenth_pair_count > eighth_pair_count:
        subdivision = 16
    else:
        subdivision = 8

    # compute mean swing ratio
    total_duration1 = 0.0
    total_duration2 = 0.0
    number_of_off_beat_hits = 0
    if subdivision == 8:
        first_off_beat_step = 2
        increment = 4
    else:
        first_off_beat_step = 1
        increment = 2
    large_8th_note_swing = False
    large_8th_note_swing_count = 0
    for step in range(first_off_beat_step, WINDOW_STEPS, increment):
        large_8th_note_swing_local = False
        if subdivision == 8 and H_flat[step] == 0 and H_flat[step + 1] == 1: # also check next step if 8th note swing since the swung note might be there instead
            step = step + 1
            large_8th_note_swing_local = True
            large_8th_note_swing_count += 1
        if H_flat[step] == 1:
            if large_8th_note_swing_local:
                prev_step = step - 3
                next_step = step + 1
            else:
                prev_step = step - first_off_beat_step
                next_step = step + first_off_beat_step
            duration1 = (step + O_flat[step]) - (prev_step + O_flat[prev_step])
            if next_step >= WINDOW_STEPS:
                duration2 = next_step - (step + O_flat[step])
            else:
                duration2 = (next_step + O_flat[next_step]) - (step + O_flat[step])
            if duration1 > 0 and duration2 > 0:
                total_duration1 += duration1
                total_duration2 += duration2
                number_of_off_beat_hits += 1

    mean_swing_ratio = (
        total_duration1 / total_duration2
        if total_duration2 > 0
        else 0
    )

    if large_8th_note_swing_count >= 4:
        large_8th_note_swing = True   

    # compute mean backbeat delay
    number_of_backbeat_hits = 0
    total_backbeat_delay = 0
    for step in range(4, WINDOW_STEPS, 8):
        if pattern_H[step, 3] == 1:  # snare hit
            total_backbeat_delay += pattern_O[step, 3]
            number_of_backbeat_hits += 1
    mean_backbeat_delay = total_backbeat_delay / number_of_backbeat_hits if number_of_backbeat_hits > 0 else 0

    mean_offset = np.mean(pattern_O[pattern_H == 1])

    # center offsets 
    if center_offsets:
        pattern_O[pattern_H == 1] -= mean_offset

    #calculate std of offsets
    std_offset = np.std(pattern_O[pattern_H == 1])

    # calculate mean velocity
    mean_velocity = np.mean(pattern_V[pattern_H == 1])

    # velocity variation 
    std_velocity = np.std(pattern_V[pattern_H == 1])

    return swing_density, mean_swing_ratio, mean_backbeat_delay, subdivision, large_8th_note_swing, mean_offset, mean_velocity, std_offset, std_velocity

# apply swing factor to a pattern by modifying the offset matrix
def apply_swing(pattern_H, pattern_O, pattern_V, swing_factor):
    swung_pattern_O = np.copy(pattern_O)
    # check if pattern is in 8th notes or 16th notes
    subdivision = "8th" if pattern_H[1::2].sum() <= 8 else "16th"

    print (pattern_H[1::2].sum() )

    if swing_factor < 0:
        quant_amount = -swing_factor
        swung_pattern_O *= (1.0 - quant_amount)

    elif swing_factor > 0:
        MAX_SWING = 0.5
        if subdivision == "8th":
            MAX_SWING = 0.75

        if subdivision == "8th":
            for step in range(2, WINDOW_STEPS, 4):  
                hit_mask = pattern_H[step] == 1
                swung_pattern_O[step, hit_mask] += swing_factor * MAX_SWING
                swung_pattern_O[step, hit_mask] = np.clip(swung_pattern_O[step, hit_mask],-MAX_SWING,MAX_SWING)
        else:  # 16th notes
            for step in range(1, WINDOW_STEPS, 2):  
                hit_mask = pattern_H[step] == 1
                swung_pattern_O[step, hit_mask] += swing_factor * MAX_SWING
                swung_pattern_O[step, hit_mask] = np.clip(swung_pattern_O[step, hit_mask],-MAX_SWING,MAX_SWING)
        print_pattern(pattern_H, swung_pattern_O, pattern_V)
        # if offset > 0.5, move hit to next step
        for step in range(WINDOW_STEPS):
            for drum_id in range(9):
                if swung_pattern_O[step, drum_id] > 0.5:
                    if step < WINDOW_STEPS - 1:
                        pattern_H[step, drum_id] = 0
                        pattern_H[step + 1, drum_id] = 1
                        swung_pattern_O[step + 1, drum_id] += swung_pattern_O[step, drum_id] - 1.0
                        swung_pattern_O[step + 1, drum_id] = np.clip(swung_pattern_O[step + 1, drum_id],-MAX_SWING,MAX_SWING)
                        pattern_V[step + 1, drum_id] = max(pattern_V[step + 1, drum_id], pattern_V[step, drum_id])
                        pattern_V[step, drum_id] = 0
    return pattern_H, swung_pattern_O, pattern_V

# print pattern arrays in a readable format for debugging
def print_pattern(pattern_H, pattern_O, pattern_V):
    for step in range(WINDOW_STEPS):
        hits = []
        for drum_id in range(9):
            if pattern_H[step, drum_id] == 1:
                offset = pattern_O[step, drum_id]
                velocity = pattern_V[step, drum_id]
                hits.append(f"{TO_MIDI_MAPPING[drum_id]}(offset={offset:.2f}, vel={velocity:.2f})")
        if hits:
            print(f"Step {step}: " + ", ".join(hits))

def reconstruct_midi(patterns_H, patterns_O, patterns_V, tempo, mean_offset):
    patterns_O = np.asarray(patterns_O).copy()
    patterns_V = np.asarray(patterns_V).copy()
    patterns_O[patterns_H == 1] += mean_offset
    patterns_O = np.clip(patterns_O, -0.5, 0.5)
    step_duration = 60.0 / tempo / SUBDIVISION_PER_BEAT
    reconstructed_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
    for step in range(WINDOW_STEPS):
        for drum_id in range(9):
            if patterns_H[step, drum_id] == 1:
                pitch = TO_MIDI_MAPPING[drum_id]
                velocity = int(patterns_V[step, drum_id] * 127)
                start_time = step * step_duration + patterns_O[step, drum_id] * step_duration
                end_time = start_time + step_duration * 0.9
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                drum_inst.notes.append(note)
    reconstructed_midi.instruments.append(drum_inst)
    return reconstructed_midi
    
def create_fingerprint(pattern_H, pattern_V):
    filtered = pattern_H * (pattern_V > 0.3)
    return filtered.flatten()

def main(dataset_path):
    dataset_path = Path(dataset_path).expanduser().resolve()
    info_path = dataset_path / "info.csv"

    if not info_path.is_file():
        raise FileNotFoundError(
            f"Could not find Groove dataset metadata at: {info_path}"
        )
    
    info = pd.read_csv(info_path)
    print(info.columns)
    # TODO: split the dataset into train, validation, and test sets based on the split column in info.csv

    split_array = []
    filename_array = []
    patterns_H_array = []
    patterns_O_array = []
    patterns_V_array = []
    tempo_array = []
    drummer_array = []
    style_array = []
    swing_density_array = []
    mean_swing_ratio_array = []
    mean_backbeat_delay_array = []
    subdivision_array = []
    fingerprint_array = []
    filename_array = []
    large_8th_note_swing_array = []
    mean_offset_array = []
    mean_velocity_array = []
    std_offset_array = []
    std_velocity_array = []
    feel_array = []
    overall_dynamic_array = []
    dynamic_variation_array = []

    pattern_num = 0

    # process all midi files in the dataset
    for index, row in info.iterrows():
        if row["beat_type"] != "beat":
            continue
        if row["time_signature"] != "4-4":
            continue

        print(f"Processing {row['midi_filename']} with tempo {row['bpm']}")

        split = row["split"]
        filename = row["midi_filename"].replace("/", "_").replace(".mid", "")
        tempo = row["bpm"]
        drummer = row["drummer"]
        style = row["style"]
        if "groove" in style:
            style = style.split("/")[0]

        midi_path = dataset_path / row["midi_filename"]
        if not midi_path.is_file():
            raise FileNotFoundError(f"Could not find MIDI file: {midi_path}")

        new_midi = remap_midi_file(str(midi_path))
        patterns_H, patterns_O, patterns_V = extract_patterns(new_midi, row["bpm"])

        for i in range(len(patterns_H)):
            pattern_num += 1
            swing_density, mean_swing_ratio, mean_backbeat_delay, subdivision, large_8th_note_swing, mean_offset, mean_velocity, std_offset, std_velocity = measure_groove_features(patterns_H[i], patterns_O[i], patterns_V[i])

            swing_density_array.append(swing_density)
            mean_swing_ratio_array.append(mean_swing_ratio)
            mean_backbeat_delay_array.append(mean_backbeat_delay)
            subdivision_array.append(subdivision)
            fingerprint = create_fingerprint(patterns_H[i], patterns_V[i])
            fingerprint_array.append(fingerprint)
            large_8th_note_swing_array.append(large_8th_note_swing)
            mean_offset_array.append(mean_offset)
            std_offset_array.append(std_offset)
            mean_velocity_array.append(mean_velocity)
            std_velocity_array.append(std_velocity)

            # these values are gotten from the quantile analysis of the dataset 
            if mean_swing_ratio >= 1.5:
                feel = "swung"
            # elif mean_swing_ratio >= 1.2 and style in ["funk", "hiphop"]:
            #     feel = "swung"
            elif std_offset >= 0.13:
                feel = "loose"
            else:
                feel = "tight"
            feel_array.append(feel)

            if mean_velocity < 0.4832396:
                overall_dynamic = "soft"
            elif mean_velocity < 0.6237068:
                overall_dynamic = "medium"
            else:
                overall_dynamic = "loud"
            overall_dynamic_array.append(overall_dynamic)

            if std_velocity < 0.20901458:
                dynamic_variation = "low"
            elif std_velocity < 0.2681045:
                dynamic_variation = "medium"
            else:
                dynamic_variation = "high"
            dynamic_variation_array.append(dynamic_variation)

        patterns_H_array += patterns_H
        patterns_O_array += patterns_O
        patterns_V_array += patterns_V
        tempo_array += [tempo] * len(patterns_H)
        style_array += [style] * len(patterns_H)
        filename_array += [filename] * len(patterns_H)
        split_array += [split] * len(patterns_H)
    
    split_dict = {"train": {}, "validation": {}, "test": {}}

    for s in split_dict.keys():
        split_dict[s]["patterns_H"] = []
        split_dict[s]["patterns_O"] = []
        split_dict[s]["patterns_V"] = []
        split_dict[s]["swing_density"] = []
        split_dict[s]["mean_swing_ratio"] = []
        split_dict[s]["mean_backbeat_delay"] = []
        split_dict[s]["subdivision"] = []
        split_dict[s]["tempo"] = []
        split_dict[s]["style"] = []
        split_dict[s]["filename"] = []
        split_dict[s]["fingerprint"] = []
        split_dict[s]["large_8th_note_swing"] = []
        split_dict[s]["mean_offset"] = []
        split_dict[s]["mean_velocity"] = []
        split_dict[s]["std_offset"] = []
        split_dict[s]["std_velocity"] = []
        split_dict[s]["feel"] = []
        split_dict[s]["overall_dynamic"] = []
        split_dict[s]["dynamic_variation"] = []

    for i in range(len(patterns_H_array)):
        # split into train, validation, and test sets based on the split column in info.csv
        split = split_array[i]
        split_dict[split]["patterns_H"].append(patterns_H_array[i])
        split_dict[split]["patterns_O"].append(patterns_O_array[i])
        split_dict[split]["patterns_V"].append(patterns_V_array[i])
        split_dict[split]["swing_density"].append(swing_density_array[i])
        split_dict[split]["mean_swing_ratio"].append(mean_swing_ratio_array[i])
        split_dict[split]["mean_backbeat_delay"].append(mean_backbeat_delay_array[i])
        split_dict[split]["subdivision"].append(subdivision_array[i])
        split_dict[split]["tempo"].append(tempo_array[i])
        split_dict[split]["style"].append(style_array[i])
        split_dict[split]["filename"].append(filename_array[i])
        split_dict[split]["fingerprint"].append(fingerprint_array[i])
        split_dict[split]["large_8th_note_swing"].append(large_8th_note_swing_array[i])
        split_dict[split]["mean_offset"].append(mean_offset_array[i])
        split_dict[split]["mean_velocity"].append(mean_velocity_array[i])
        split_dict[split]["std_offset"].append(std_offset_array[i])
        split_dict[split]["std_velocity"].append(std_velocity_array[i])
        split_dict[split]["feel"].append(feel_array[i])
        split_dict[split]["overall_dynamic"].append(overall_dynamic_array[i])
        split_dict[split]["dynamic_variation"].append(dynamic_variation_array[i])

    # save to h5py
    with h5py.File("processed_groove_dataset_split.h5", "w") as f:
        f.create_group("train")
        f.create_group("validation")
        f.create_group("test")

        for s in split_dict.keys():
            f[s].create_dataset("patterns_H", data=np.array(split_dict[s]["patterns_H"]))
            f[s].create_dataset("patterns_O", data=np.array(split_dict[s]["patterns_O"]))
            f[s].create_dataset("patterns_V", data=np.array(split_dict[s]["patterns_V"]))
            f[s].create_dataset("swing_density", data=np.array(split_dict[s]["swing_density"]))
            f[s].create_dataset("mean_swing_ratio", data=np.array(split_dict[s]["mean_swing_ratio"]))
            f[s].create_dataset("mean_backbeat_delay", data=np.array(split_dict[s]["mean_backbeat_delay"]))
            f[s].create_dataset("subdivision", data=np.array(split_dict[s]["subdivision"]))
            f[s].create_dataset("tempo", data=np.array(split_dict[s]["tempo"]))
            f[s].create_dataset("style", data=np.array(split_dict[s]["style"], dtype='S'))
            f[s].create_dataset("filename", data=np.array(split_dict[s]["filename"], dtype='S'))
            f[s].create_dataset("fingerprint", data=np.array(split_dict[s]["fingerprint"]))
            f[s].create_dataset("large_8th_note_swing", data=np.array(split_dict[s]["large_8th_note_swing"]))
            f[s].create_dataset("mean_offset", data=np.array(split_dict[s]["mean_offset"]))
            f[s].create_dataset("mean_velocity", data=np.array(split_dict[s]["mean_velocity"]))
            f[s].create_dataset("std_offset", data=np.array(split_dict[s]["std_offset"]))
            f[s].create_dataset("std_velocity", data=np.array(split_dict[s]["std_velocity"]))
            f[s].create_dataset("feel", data=np.array(split_dict[s]["feel"], dtype='S'))
            f[s].create_dataset("overall_dynamic", data=np.array(split_dict[s]["overall_dynamic"], dtype='S'))
            f[s].create_dataset("dynamic_variation", data=np.array(split_dict[s]["dynamic_variation"], dtype='S'))
    # reconstruct all patterns and save to midi for verification
    # for i in range(len(patterns_H)):
    #     reconstructed_midi = reconstruct_midi(patterns_H[i], patterns_O[i], patterns_V[i], tempo)
    #     reconstructed_midi.write(f"groove_processed/{original_filename}_reconstructed_{i}.mid")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the Groove MIDI Dataset into two-bar patterns."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the extracted Groove dataset directory containing info.csv.",
    )
    args = parser.parse_args()
    main(args.dataset_path)
