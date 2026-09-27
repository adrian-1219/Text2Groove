# This script creates the baseline midi files using the genre-labelled groove templates found in Ableton and Reason.
# Also creates the gmd nearest neighbour midi files using the gmd dataset 

# 1. Process the groove template midi files and calculate groove features
# 2. Map each possible prompt to a groove template midi 
# 3. Apply prompt groove template midi files to the sample quantized patterns

# 4. Find nearest neighbour groove from the gmd dataset and apply it to the sample quantized pattern

import os
import struct
import pretty_midi
import numpy as np
from midi_data_preprocessing import reconstruct_midi, extract_patterns
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from generate_training_pairs import map_offsets
from midi_data_preprocessing import measure_groove_features

SIMILARITY_THRESHOLD = 0.67

WINDOW_SIZE = 2 # 2 bars per window
HOP_SIZE = 1 # 1 bar hop size
SUBDIVISION_PER_BEAT = 4
BEATS_PER_BAR = 4

STEPS_PER_BAR = BEATS_PER_BAR * SUBDIVISION_PER_BEAT
WINDOW_STEPS = WINDOW_SIZE * STEPS_PER_BAR
HOP_STEP = HOP_SIZE * STEPS_PER_BAR

STRAIGHT_THRESHOLD = 0.07
LOOSE_THRESHOLD = 0.13
SOFT_THRESHOLD = 0.4832396
LOUD_THRESHOLD = 0.6237068

DEBUG = False

def pattern_key(pattern: np.ndarray, tempo: float) -> bytes:
    pattern = np.asarray(pattern, dtype=np.uint8)
    pattern_bytes = np.packbits(pattern.reshape(-1)).tobytes()
    tempo_bytes = struct.pack("<f", float(tempo))

    return pattern_bytes + tempo_bytes

def extract_pattern_1d(midi, tempo):
    seconds_per_beat = 60.0 / tempo
    step_duration = seconds_per_beat / SUBDIVISION_PER_BEAT

    H = np.zeros((WINDOW_STEPS, 9), dtype=np.float32)
    O = np.zeros((WINDOW_STEPS, 9), dtype=np.float32)
    V = np.zeros((WINDOW_STEPS, 9), dtype=np.float32)

    # Combine notes from all instruments into the first window
    for inst in midi.instruments:
        for note in inst.notes:
            global_t = note.start / step_duration
            global_step = int(np.round(global_t))

            # First window starts at step 0
            local_step = global_step

            if not 0 <= local_step < WINDOW_STEPS:
                continue

            H[local_step, 0] = 1.0

            offset = global_t - global_step
            velocity = note.velocity / 127.0

            # Keep the loudest note if multiple notes occupy the same step
            if velocity > V[local_step, 0]:
                O[local_step, 0] = offset
                V[local_step, 0] = velocity

    return H, O, V


def measure_groove_features_lite(pattern_H, pattern_O, pattern_V):
    # consolidate the matrices to 1D arrays 
    H_flat = pattern_H[:, 0]
    O_flat = pattern_O[:, 0]
    V_flat = pattern_V[:, 0]

    #calculate std of offsets
    std_offset = np.std(O_flat[H_flat == 1])

    # calculate mean velocity
    mean_velocity = np.mean(V_flat[H_flat == 1])

    return mean_velocity, std_offset

def get_mean_swing_ratio(pattern_H, pattern_O, subdivision):
    H_flat = pattern_H[:, 0]
    O_flat = pattern_O[:, 0]
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
    return mean_swing_ratio

def map_offsets_and_velocities(pattern_H, pattern_O, pattern_V, new_pattern_H):
    new_pattern_O = np.zeros_like(new_pattern_H)
    new_pattern_V = np.zeros_like(new_pattern_H)

    H_flat = pattern_H[:, 0]
    O_flat = pattern_O[:, 0]
    V_flat = pattern_V[:, 0]

    for step in range(WINDOW_STEPS):
        for drum_id in range(9):
            if new_pattern_H[step, drum_id] == 1:
                if H_flat[step] == 1:
                    new_pattern_O[step, drum_id] = O_flat[step]
                    new_pattern_V[step, drum_id] = V_flat[step]
                else:
                    # otherwise set the offset to 0 and velocity to the default velocity of 0.75
                    new_pattern_O[step, drum_id] = 0
                    new_pattern_V[step, drum_id] = 0.75


    return new_pattern_O, new_pattern_V

def main():
    mean_swing_ratios = []
    mean_velocities = []
    std_offsets = []
    subdivisions = []
    file_names = []
    genres = []
    patterns_O = []
    patterns_V = []
    patterns_H = []
    count = 0

    TEMPO = 120 # all groove templates were exported in 120 bpm

    # Stimuli set 1: Groove template midi files
    # 1. Go through all groove template midi files in the Ableton and Reason folders and calculate groove features
    folder_paths = ["Groove Templates/Ableton", "Groove Templates/Reason"]
    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            if file.endswith(".mid"):
                file_path = os.path.join(folder_path, file)
                midi_data = pretty_midi.PrettyMIDI(file_path)

                # genre
                if "funk" in file.lower():
                    genre = "funk"
                elif "rock" in file.lower():
                    genre = "rock"
                elif "jazz" in file.lower():
                    genre = "jazz"
                elif "hiphop" in file.lower() or "hip hop" in file.lower():
                    genre = "hiphop"
                else:
                    genre = "unknown"
                
                # subdivision
                if "8ths" in file.lower():
                    subdivision = 8
                else:
                    subdivision = 16
                
                # file name
                file_name = file.split(".")[0]

                # extract patterns
                midi_data = pretty_midi.PrettyMIDI(file_path)
                pattern_H, pattern_O, pattern_V = extract_pattern_1d(midi_data, TEMPO)

                if DEBUG:
                    # test if the midi was processed correctly 
                    reconstructed_midi = reconstruct_midi(pattern_H, pattern_O, pattern_V, TEMPO, mean_offset=0)
                    reconstructed_midi.write(f"groove_processed/{file_name}.mid")
                    print("Wrote midi file to groove_processed folder for verification:", file_name)

                # measure groove features
                mean_velocity, std_offset = measure_groove_features_lite(pattern_H, pattern_O, pattern_V)
                mean_swing_ratio_value = get_mean_swing_ratio(pattern_H, pattern_O, subdivision)

                if DEBUG:
                    print(f"File: {file_name}, Genre: {genre}, Subdivision: {subdivision}, Mean Swing Ratio: {mean_swing_ratio_value:.2f}, Mean Velocity: {mean_velocity:.2f}, Std Offset: {std_offset:.2f}")
                    count += 1
                    if count == 5:
                        quit()

                genres.append(genre)
                subdivisions.append(subdivision)
                file_names.append(file_name)
                mean_swing_ratios.append(mean_swing_ratio_value)
                mean_velocities.append(mean_velocity)
                std_offsets.append(std_offset)
                patterns_H.append(pattern_H)
                patterns_O.append(pattern_O)
                patterns_V.append(pattern_V)

    # 2. Map each possible prompt to a groove template midi 
    GENRES = [
        "funk",
        "rock",
        "jazz",
        "hiphop",
    ]

    FEEL = [
        "swung",
        "tight",
        "loose",
    ]

    DYNAMICS = [
        "soft",
        "medium",
        "loud"
    ]

    # required combinations dictonary 
    required_combinations = {
        (genre, feel, dynamic): None
        for genre in GENRES
        for feel in FEEL
        for dynamic in DYNAMICS
    }

    used_templates = set()

    
    for combination in required_combinations:
        genre, feel, dynamic = combination
        for i in range(len(file_names)):
            if i in used_templates:
                continue
            if genres[i] == genre:
                if (mean_swing_ratios[i] >= 1.5 or "swing" in file_names[i].lower() or "swung" in file_names[i].lower()):
                    if feel != "swung":
                        continue
                    if dynamic == "soft" and mean_velocities[i] < SOFT_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                    elif dynamic == "medium" and SOFT_THRESHOLD <= mean_velocities[i] < LOUD_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                    elif dynamic == "loud" and mean_velocities[i] >= LOUD_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                elif mean_swing_ratios[i] >= 1.17:
                    if feel != "loose":
                        continue
                    if dynamic == "soft" and mean_velocities[i] < SOFT_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                    elif dynamic == "medium" and SOFT_THRESHOLD <= mean_velocities[i] < LOUD_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                    elif dynamic == "loud" and mean_velocities[i] >= LOUD_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                else:
                    if feel != "tight":
                        continue
                    if dynamic == "soft" and mean_velocities[i] < SOFT_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                    elif dynamic == "medium" and SOFT_THRESHOLD <= mean_velocities[i] < LOUD_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
                    elif dynamic == "loud" and mean_velocities[i] >= LOUD_THRESHOLD:
                        required_combinations[combination] = i
                        used_templates.add(i)
                        break
    
    # for each required combination that was not found, find another groove template but diregard dynamic level
    for combination in required_combinations:
        print(combination, required_combinations[combination])
        if required_combinations[combination] is None:
            genre, feel, dynamic = combination
            for i in range(len(file_names)):
                if i in used_templates:
                    continue
                if genres[i] == genre:
                    if (mean_swing_ratios[i] >= 1.5 or "swing" in file_names[i].lower() or "swung" in file_names[i].lower()):
                        if feel != "swung":
                            continue
                        print(f"Found groove template for combination: {combination}, file name: {file_names[i]}, mean_velocity: {mean_velocities[i]:.2f}")
                        used_templates.add(i)
                        break
                    elif mean_swing_ratios[i] >= 1.17:
                        if feel != "loose":
                            continue
                        print(f"Found groove template for combination: {combination}, file name: {file_names[i]}, mean_velocity: {mean_velocities[i]:.2f}")
                        used_templates.add(i)
                        break
                    else:
                        if feel != "tight":
                            continue
                        print(f"Found groove template for combination: {combination}, file name: {file_names[i]}, mean_velocity: {mean_velocities[i]:.2f}")
                        used_templates.add(i)
                        break


    # # print combination where groove template midi files were not found
    for combination, index in required_combinations.items():
        if index is None:
            print(f"No groove template midi file found for combination: {combination}")

    # # print filenames and groove features of the groove template midi files 
    # for i in range(len(file_names)):
    #     if genres[i] == "jazz":
    #         print(f"File: {file_names[i]}, Genre: {genres[i]}, Subdivision: {subdivisions[i]}, Mean Swing Ratio: {mean_swing_ratios[i]:.2f}, Mean Velocity: {mean_velocities[i]:.2f}, Std Offset: {std_offsets[i]:.2f}")
        
    # print all soft groove template midi files and their mean velocity
    for i in range(len(file_names)):
        if mean_velocities[i] < SOFT_THRESHOLD:
            while mean_velocities[i] < 0.4:
                patterns_V[i] = patterns_V[i] * 1.1
                mean_velocities[i] = np.mean(patterns_V[i][patterns_H[i] == 1])

            print(f"File: {file_names[i]}, Genre: {genres[i]}, Subdivision: {subdivisions[i]}, Mean Swing Ratio: {mean_swing_ratios[i]:.2f}, Mean Velocity: {mean_velocities[i]:.2f}, Std Offset: {std_offsets[i]:.2f}")

    # 3. Apply prompt groove template midi files to the sample quantized patterns in the quantized folder
    quantized_patterns_H = []
    quantized_tempo = []
    quantized_filename = []

    for file in os.listdir("Groove Templates/Quantized"):
        if file.endswith(".mid"):
            file_path = os.path.join("Groove Templates/Quantized", file)
            midi_data = pretty_midi.PrettyMIDI(file_path)
            tempo_value = midi_data.get_tempo_changes()[1][0]
            tempo_value = round(tempo_value)  # round to nearest integer

            # extract patterns
            new_patterns_H, _, _ = extract_patterns(midi_data, tempo_value)
            new_pattern_H = new_patterns_H[0]  # only take the first window 
            quantized_patterns_H.append(new_pattern_H)
            quantized_tempo.append(tempo_value)
            quantized_filename.append(file)

            for combination in required_combinations:
                # apply the groove template midi file to the sample quantized pattern
                genre, feel, dynamic = combination
                groove_template_index = required_combinations[combination]
                new_pattern_O, new_pattern_V = map_offsets_and_velocities(patterns_H[groove_template_index], patterns_O[groove_template_index], patterns_V[groove_template_index], new_pattern_H)
                reconstructed_midi = reconstruct_midi(new_pattern_H, new_pattern_O, new_pattern_V, tempo_value, mean_offset=0)
                reconstructed_midi.write(f"Groove Templates/GrooveTemplateApplied/{file.split('.')[0]}_{genre}_{feel}_{dynamic}.mid")

    # Stimuli set 2: GMD nearest neighbour
    # 4. Find nearest neighbour groove from the gmd dataset and apply it to the sample quantized pattern
    # consolidate hit pattern matrix into 1 dimension
    with h5py.File("processed_groove_dataset_split.h5", "r") as f:
        # Access the datasets
        patterns_H = np.concatenate([f["train"]["patterns_H"][:], f["validation"]["patterns_H"][:], f["test"]["patterns_H"][:]])
        patterns_O = np.concatenate([f["train"]["patterns_O"][:], f["validation"]["patterns_O"][:], f["test"]["patterns_O"][:]])
        patterns_V = np.concatenate([f["train"]["patterns_V"][:], f["validation"]["patterns_V"][:], f["test"]["patterns_V"][:]])
        swing_density = np.concatenate([f["train"]["swing_density"][:], f["validation"]["swing_density"][:], f["test"]["swing_density"][:]])
        mean_swing_ratio = np.concatenate([f["train"]["mean_swing_ratio"][:], f["validation"]["mean_swing_ratio"][:], f["test"]["mean_swing_ratio"][:]])
        mean_backbeat_delay = np.concatenate([f["train"]["mean_backbeat_delay"][:], f["validation"]["mean_backbeat_delay"][:], f["test"]["mean_backbeat_delay"][:]])
        subdivision = np.concatenate([f["train"]["subdivision"][:], f["validation"]["subdivision"][:], f["test"]["subdivision"][:]])
        tempo = np.concatenate([f["train"]["tempo"][:], f["validation"]["tempo"][:], f["test"]["tempo"][:]])
        style = np.concatenate([f["train"]["style"][:], f["validation"]["style"][:], f["test"]["style"][:]])
        filename = np.concatenate([f["train"]["filename"][:], f["validation"]["filename"][:], f["test"]["filename"][:]])
        large_8th_note_swing = np.concatenate([f["train"]["large_8th_note_swing"][:], f["validation"]["large_8th_note_swing"][:], f["test"]["large_8th_note_swing"][:]])
        std_offset = np.concatenate([f["train"]["std_offset"][:], f["validation"]["std_offset"][:], f["test"]["std_offset"][:]])
        feel = np.concatenate([f["train"]["feel"][:], f["validation"]["feel"][:], f["test"]["feel"][:]])
        overall_dynamic = np.concatenate([f["train"]["overall_dynamic"][:], f["validation"]["overall_dynamic"][:], f["test"]["overall_dynamic"][:]])
        dynamic_variation = np.concatenate([f["train"]["dynamic_variation"][:], f["validation"]["dynamic_variation"][:], f["test"]["dynamic_variation"][:]])

        patterns_H_flat = np.zeros((len(patterns_H), 32))
        for i, p in enumerate(patterns_H):
            for step in range(32):
                for drum_id in range(9):
                    if p[step, drum_id] == 1:
                        patterns_H_flat[i, step] = 1
                        break 

        quantized_patterns_H_flat = np.zeros((len(quantized_patterns_H), 32))
        for i, p in enumerate(quantized_patterns_H):
            for step in range(32):
                for drum_id in range(9):
                    if p[step, drum_id] == 1:
                        quantized_patterns_H_flat[i, step] = 1
                        break

        similarity_matrix = cosine_similarity(quantized_patterns_H_flat, patterns_H_flat)

        # keep track of hit patterns that have already been used as source patterns to avoid duplicates
        used_source_patterns = set()

        for i in range(len(quantized_patterns_H)):
            pattern_number = quantized_filename[i].split("_")[1]
            print(f"Processing pattern pattern {i}")

            pattern_hash = pattern_key(quantized_patterns_H[i], quantized_tempo[i])
            if pattern_hash in used_source_patterns:
                continue
            used_source_patterns.add(pattern_hash)

            zero_offsets = np.zeros((32, 9))
            constant_velocities = zero_offsets + 0.75

            reconstructed_midi = reconstruct_midi(quantized_patterns_H[i], zero_offsets, constant_velocities, quantized_tempo[i], mean_offset=0)
            reconstructed_midi.write(f"gmd_nearest_neighbour_stimuli/debug/pattern_{i}_quantized_nearest_neighbour.mid")

            # find similar patterns based on cosine similarity of the hit patterns and groove features
            scores = []
            if DEBUG:
                print(f"Pattern {i}: Filename = {filename[i].decode('utf-8')}, Tempo = {tempo[i]}, Style = {style[i].decode('utf-8')}, Swing Density = {swing_density[i]}, Mean Swing Ratio = {mean_swing_ratio[i]}, Mean Backbeat Delay = {mean_backbeat_delay[i]}, Subdivision = {subdivision[i]}, Large 8th Note Swing = {large_8th_note_swing[i]}")

            for j in range(len(patterns_H)):
                hit_similarity = similarity_matrix[i, j]
                tempo_difference = abs(float(quantized_tempo[i]) - float(tempo[j]))

                tempo_similarity = np.exp(-tempo_difference / 30.0)

                # Control how important each component is
                HIT_WEIGHT = 0.8
                TEMPO_WEIGHT = 0.2

                combined_similarity = (
                    HIT_WEIGHT * hit_similarity
                    + TEMPO_WEIGHT * tempo_similarity
                )

                if combined_similarity > SIMILARITY_THRESHOLD:  # Only consider patterns with high similarity
                    scores.append((j, combined_similarity))
                # if style[j].decode("utf-8") == "jazz" and combined_similarity > JAZZ_SIMILARITY_THRESHOLD:
                #     scores.append((j, combined_similarity))

            scores.sort(key=lambda x: x[1], reverse=True)

            style_found = {
                "jazz": False,
                "funk": False,
                "hiphop": False,
                "rock": False,
            }
            feel_found = {
                "swung": False,
                "tight": False,
                "loose": False,
            }
            overall_dynamic_found = {
                "soft": False,
                "medium": False,
                "loud": False,
            }
            # dynamic_variation_found = {
            #     "low": False,
            #     "medium": False,
            #     "high": False,
            # }

            required_combinations = {
                (target_style, target_feel, target_overall_dynamic)
                for target_style in style_found
                for target_feel in feel_found
                for target_overall_dynamic in overall_dynamic_found
                # for target_dynamic_variation in dynamic_variation_found
            }

        
            # required_combinations.discard(("funk", "swung", "soft"))
            # required_combinations.discard(("funk", "swung", "loud"))
            # required_combinations.discard(("hiphop", "swung", "soft"))
            # required_combinations.discard(("hiphop", "swung", "loud"))

            found_combinations = set()
            found_hov = dict()  # to store the found H, O, V for each combination
            for combination in required_combinations:
                found_hov[combination] = None

            for k in range(len(scores)):
                if len(found_combinations) == len(required_combinations):
                    break
                j, similarity = scores[k]

                target_style = style[j].decode("utf-8")
                if "jazz" in target_style:
                    target_style = "jazz"
                elif "funk" in target_style:
                    target_style = "funk"

                # if target_style.split("/")[0] in style_found:
                #     target_style = target_style.split("/")[0]
                
                # if (target_style == "jazz" or target_style == "rock") and mean_swing_ratio[j] > MEAN_SWING_RATIO_THRESHOLD:

                target_feel = feel[j].decode("utf-8")
                target_overall_dynamic = overall_dynamic[j].decode("utf-8")
                target_dynamic_variation = dynamic_variation[j].decode("utf-8")

                if (target_style, target_feel, target_overall_dynamic) not in required_combinations:
                    continue
                if (target_style, target_feel, target_overall_dynamic) in found_combinations:
                    continue
                if large_8th_note_swing[j] == 1: # ignore patterns with large 8th note swing since the model does not account for note shifts
                    continue

                if DEBUG:
                    print(f"Pattern {j}: Similarity = {similarity}, Filename = {filename[j].decode('utf-8')}, Tempo = {tempo[j]}, Style = {style[j].decode('utf-8')}, Swing Density = {swing_density[j]}, Mean Swing Ratio = {mean_swing_ratio[j]}, Mean Backbeat Delay = {mean_backbeat_delay[j]}, Subdivision = {subdivision[j]}, Large 8th Note Swing = {large_8th_note_swing[j]}")

                # shift offbeat hits if large 8th note swing is detected in the patterns
                # H_j_shifted, O_j_shifted, V_j_shifted = shift_offbeat_hits(patterns_H[j].copy(), patterns_O[j].copy(), patterns_V[j].copy(), large_8th_note_swing[i], large_8th_note_swing[j])

                # map pattern j offsets to pattern i offsets depending on where the hits are in the patterns
                new_H, new_O, new_V = map_offsets(quantized_patterns_H[i], patterns_H[j], patterns_O[j], patterns_V[j])

                # check if the new pattern still satisfies constraints for the target feel
                new_swing_density, new_mean_swing_ratio, new_mean_backbeat_delay, new_subdivision, new_large_8th_note_swing, new_mean_offset, new_mean_velocity, new_std_offset, new_std_velocity = measure_groove_features(new_H, new_O, new_V)

                if target_feel == "swing" and new_mean_swing_ratio < 1.5:
                    continue
                if target_feel == "tight" and new_std_offset > STRAIGHT_THRESHOLD:
                    continue
                if target_feel == "loose" and new_std_offset < LOOSE_THRESHOLD:
                    continue

                if new_mean_velocity < SOFT_THRESHOLD:
                    target_overall_dynamic = "soft"
                elif new_mean_velocity < LOUD_THRESHOLD:
                    target_overall_dynamic = "medium"
                else:
                    target_overall_dynamic = "loud"

                # if target_overall_dynamic == "soft" and new_mean_velocity > SOFT_THRESHOLD:
                #     continue
                # if target_overall_dynamic == "medium" and (new_mean_velocity < SOFT_THRESHOLD or new_mean_velocity > LOUD_THRESHOLD):
                #     continue
                # if target_overall_dynamic == "loud" and new_mean_velocity < LOUD_THRESHOLD:
                #     continue


                found_combinations.add((target_style, target_feel, target_overall_dynamic))
                found_hov[(target_style, target_feel, target_overall_dynamic)] = (quantized_patterns_H[i], new_O, new_V)

                reconstructed_midi = reconstruct_midi(quantized_patterns_H[i], new_O, new_V, quantized_tempo[i], mean_offset=0)
                reconstructed_midi.write(f"gmd_nearest_neighbour_stimuli/midi/pattern_{pattern_number}_nearest_neighbour_{target_style}_{target_feel}_{target_overall_dynamic}.mid")

                # if DEBUG:
                #     reconstruct_midi_patterns(i, j, patterns_H[i], patterns_O[i], patterns_V[i], patterns_H[j], patterns_O[j], patterns_V[j], patterns_H[i], new_O, new_V, tempo[i], instruction)
            
            for combination in required_combinations:
                if combination not in found_combinations:
                    print(f"Pattern {pattern_number}: Could not find a nearest neighbour for combination: {combination}")
                    g, f, d = combination

                    if found_hov[(g,f,"soft")] is not None:
                        existing_H, existing_O, existing_V = found_hov[(g,f,"soft")]
                        existing_dynamic = "soft"
                    elif found_hov[(g,f,"medium")] is not None:
                        existing_H, existing_O, existing_V = found_hov[(g,f,"medium")]
                        existing_dynamic = "medium"
                    elif found_hov[(g,f,"loud")] is not None:
                        existing_H, existing_O, existing_V = found_hov[(g,f,"loud")]
                        existing_dynamic = "loud"
                    else:
                        print(f"Pattern {pattern_number}: NONE FOUND for combination: {combination}")
                        continue

                    # reconstruct midi with the existing combination
                    # if d == soft and existing dynamic is medium or loud, then scale down the velocities to be soft
                    if d == "soft" and existing_dynamic in ["medium", "loud"]:
                        while np.mean(existing_V[existing_H == 1]) > SOFT_THRESHOLD:
                            existing_V = existing_V * 0.9
                            existing_V[existing_V > 1.0] = 1.0  # clip to max velocity of 1.0
                        new_V = existing_V
                    elif d == "medium" and existing_dynamic in ["soft", "loud"]:
                        while np.mean(existing_V[existing_H == 1]) < SOFT_THRESHOLD or np.mean(existing_V[existing_H == 1]) > LOUD_THRESHOLD:
                            existing_V = existing_V * 1.1 if np.mean(existing_V[existing_H == 1]) < SOFT_THRESHOLD else existing_V * 0.9
                            existing_V[existing_V > 1.0] = 1.0  # clip to max velocity of 1.0
                        new_V = existing_V
                    elif d == "loud" and existing_dynamic in ["soft", "medium"]:
                        while np.mean(existing_V[existing_H == 1]) < LOUD_THRESHOLD:
                            existing_V = existing_V * 1.1
                            existing_V[existing_V > 1.0] = 1.0  # clip to max velocity of 1.0
                        new_V = existing_V

                    found_combinations.add((g, f, d))
                    found_hov[(g, f, d)] = (quantized_patterns_H[i], existing_O, new_V)
                    reconstructed_midi = reconstruct_midi(quantized_patterns_H[i], existing_O, new_V, quantized_tempo[i], mean_offset=0)
                    reconstructed_midi.write(f"gmd_nearest_neighbour_stimuli/midi/pattern_{pattern_number}_nearest_neighbour_{g}_{f}_{d}.mid")
                    

            for combination in required_combinations:
                if combination not in found_combinations:
                    print(f"Pattern {pattern_number}: STILL Could not find a nearest neighbour for combination: {combination}")
                        





                

if __name__ == "__main__":
    main()
            
