import pretty_midi
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from midi_data_preprocessing import reconstruct_midi
import numpy as np
from midi_data_preprocessing import measure_groove_features
import struct

DEBUG = False
TEMPO_WINDOW = 50
SWING_DENSITY_THRESHOLD = 0.15
MEAN_SWING_RATIO_THRESHOLD = 1.5
FUNK_HIPHOP_MEAN_SWING_RATIO_THRESHOLD = 1.2
STRAIGHT_THRESHOLD = 0.1
LOOSE_THRESHOLD = 0.13
SIMILARITY_THRESHOLD = 0.85
SOFT_THRESHOLD = 0.4832396
LOUD_THRESHOLD = 0.6237068

SOURCE_HITS = []
SOURCE_OFFSETS = []
SOURCE_VELOCITIES = []

TARGET_HITS = []
TARGET_OFFSETS = []
TARGET_VELOCITIES = []

STYLE = []
FEEL = []
DYNAMIC = []
VARIATION = []
TEMPOS = []

DELTA_OFFSETS = []
DELTA_VELOCITIES = []

# map pattern j offsets to pattern i offsets depending on where the hits are in the patterns
def map_offsets(H_i, H_j, O_j, V_j, default_velocity=0.15):
    new_H = H_i.copy()
    new_O = np.zeros_like(O_j)
    new_V = np.zeros_like(V_j)
    new_V[new_H == 1] = default_velocity

    for step in range(len(new_H)):
        original_donors = list(np.flatnonzero(H_j[step]))

        if not original_donors:
            continue

        available_donors = original_donors.copy()

        for drum_id in np.flatnonzero(new_H[step]):
            if not available_donors:
                available_donors = original_donors.copy()

            closest_drum_id = min(
                available_donors,
                key=lambda donor_id: abs(donor_id - drum_id),
            )

            new_O[step, drum_id] = O_j[step, closest_drum_id]
            new_V[step, drum_id] = V_j[step, closest_drum_id]
            available_donors.remove(closest_drum_id)

    return new_H, new_O, new_V

# shift offbeat hits if large 8th note swing is detected in the patterns
def shift_offbeat_hits(H_j, O_j, V_j, large_8th_note_swing_i, large_8th_note_swing_j):
    if not large_8th_note_swing_i and large_8th_note_swing_j:
        for step in range(2, 32, 4):
            for drum_id in range(len(H_j[step])):
                if H_j[step][drum_id] == 0 and H_j[step + 1][drum_id] == 1:
                    # shift offbeat hits earlier by 1 step and increase offset value
                    H_j[step][drum_id] = H_j[step + 1][drum_id]
                    H_j[step + 1][drum_id] = 0
                    O_j[step][drum_id] = O_j[step + 1][drum_id] + 1
                    O_j[step + 1][drum_id] = 0
                    V_j[step][drum_id] = V_j[step + 1][drum_id]
                    V_j[step + 1][drum_id] = 0
    return H_j, O_j, V_j

# reconstructuct the pattern i, pattern j and the transformed pattern to midi files for debugging 
def reconstruct_midi_patterns(i, j, H_i, O_i, V_i, H_j, O_j, V_j, new_H, new_O, new_V, tempo, instruction):
    mean_offset = 0
    # reconstruct the old pattern to compare with the new pattern
    old_midi = reconstruct_midi(H_i, O_i, V_i, tempo, mean_offset)
    old_midi.write(f"groove_processed/pattern_{i}_original.mid")
    # reconstruct the compared pattern for debugging 
    compared_midi = reconstruct_midi(H_j, O_j, V_j, tempo, mean_offset)
    compared_midi.write(f"groove_processed/pattern_{j}_compared_{instruction}.mid")
    # reconstruct the new pattern and save to midi
    new_midi = reconstruct_midi(new_H, new_O, new_V, tempo, mean_offset)
    new_midi.write(f"groove_processed/pattern_{i}_from_pattern_{j}_{instruction}.mid")

# append training pairs to arrays
def add_training_pair(source_H, source_O, source_V, target_H, target_O, target_V, style, feel, dynamic, tempo):
    SOURCE_HITS.append(source_H)
    SOURCE_OFFSETS.append(source_O)
    SOURCE_VELOCITIES.append(source_V)

    TARGET_HITS.append(target_H)
    TARGET_OFFSETS.append(target_O)
    TARGET_VELOCITIES.append(target_V)

    STYLE.append(style)
    FEEL.append(feel)
    DYNAMIC.append(dynamic)
    # VARIATION.append(variation)
    TEMPOS.append(tempo)

    delta_O = target_O - source_O
    delta_V = target_V - source_V
    DELTA_OFFSETS.append(delta_O)
    DELTA_VELOCITIES.append(delta_V)

def pattern_key(pattern: np.ndarray, tempo: float) -> bytes:
    pattern = np.asarray(pattern, dtype=np.uint8)
    pattern_bytes = np.packbits(pattern.reshape(-1)).tobytes()
    tempo_bytes = struct.pack("<f", float(tempo))

    return pattern_bytes + tempo_bytes

def main():

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

        # consolidate hit pattern matrix into 1 dimension
        patterns_H_flat = np.zeros((len(patterns_H), 32))
        for i, p in enumerate(patterns_H):
            for step in range(32):
                for drum_id in range(9):
                    if p[step, drum_id] == 1:
                        patterns_H_flat[i, step] = 1
                        break 


        similarity_matrix = cosine_similarity(patterns_H_flat)

        # # compute similarity matrix for all patterns 
        # num_patterns = len(patterns_H)
        # flat_patterns = patterns_H.reshape(num_patterns, -1)
        # similarity_matrix = cosine_similarity(flat_patterns)

        # keep track of hit patterns that have already been used as source patterns to avoid duplicates
        used_source_patterns = set()


        for i in range(len(patterns_H)):
            if large_8th_note_swing[i] == 1: # ignore patterns with large 8th note swing since the model does not account for note shifts
                continue
            print(f"Processing pattern {i+1}/{len(patterns_H)}")

            pattern_hash = pattern_key(patterns_H[i], tempo[i])
            if pattern_hash in used_source_patterns:
                continue
            used_source_patterns.add(pattern_hash)

            zero_offsets = np.zeros((32, 9))
            constant_velocities = zero_offsets + 0.75

            # find similar patterns based on cosine similarity of the hit patterns and groove features
            scores = []
            if DEBUG:
                print(f"Pattern {i}: Filename = {filename[i].decode('utf-8')}, Tempo = {tempo[i]}, Style = {style[i].decode('utf-8')}, Swing Density = {swing_density[i]}, Mean Swing Ratio = {mean_swing_ratio[i]}, Mean Backbeat Delay = {mean_backbeat_delay[i]}, Subdivision = {subdivision[i]}, Large 8th Note Swing = {large_8th_note_swing[i]}")

            for j in range(len(patterns_H)):
                if i == j:
                    continue
                hit_similarity = similarity_matrix[i, j]
                tempo_difference = abs(float(tempo[i]) - float(tempo[j]))

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
                # "medium": False,
                "loud": False,
            }
            # dynamic_variation_found = {
            #     "low": False,
            #     "medium": False,
            #     "high": False,
            # }

            required_combinations = {
                (target_style, target_feel)
                for target_style in style_found
                for target_feel in feel_found
                # for target_overall_dynamic in overall_dynamic_found
                # for target_dynamic_variation in dynamic_variation_found
            }

    

            found_combinations = set()

            for k in range(len(scores)):
                if len(found_combinations) == len(required_combinations):
                    break
                j, similarity = scores[k]

                target_style = style[j].decode("utf-8")
                if "jazz" in target_style:
                    target_style = "jazz"
                elif "funk" in target_style:
                    target_style = "funk"

                target_feel = feel[j].decode("utf-8")
                if target_style == "funk" or target_style == "hiphop":
                    if mean_swing_ratio[j] >= FUNK_HIPHOP_MEAN_SWING_RATIO_THRESHOLD:
                        target_feel = "swung"


                target_overall_dynamic = overall_dynamic[j].decode("utf-8")
                target_dynamic_variation = dynamic_variation[j].decode("utf-8")

                if (target_style, target_feel) not in required_combinations:
                    continue
                if (target_style, target_feel) in found_combinations:
                    continue

                if DEBUG:
                    print(f"Pattern {j}: Similarity = {similarity}, Filename = {filename[j].decode('utf-8')}, Tempo = {tempo[j]}, Style = {style[j].decode('utf-8')}, Swing Density = {swing_density[j]}, Mean Swing Ratio = {mean_swing_ratio[j]}, Mean Backbeat Delay = {mean_backbeat_delay[j]}, Subdivision = {subdivision[j]}, Large 8th Note Swing = {large_8th_note_swing[j]}")

                # map pattern j offsets to pattern i offsets depending on where the hits are in the patterns
                new_H, new_O, new_V = map_offsets(patterns_H[i], patterns_H[j], patterns_O[j], patterns_V[j])

                # check if the new pattern still satisfies constraints for the target feel
                new_swing_density, new_mean_swing_ratio, new_mean_backbeat_delay, new_subdivision, new_large_8th_note_swing, new_mean_offset, new_mean_velocity, new_std_offset, new_std_velocity = measure_groove_features(new_H, new_O, new_V)

                if target_style == "funk" or target_style == "hiphop":
                    if target_feel == "swung" and new_mean_swing_ratio < FUNK_HIPHOP_MEAN_SWING_RATIO_THRESHOLD:
                        continue
                else:
                    if target_feel == "swung" and new_mean_swing_ratio < MEAN_SWING_RATIO_THRESHOLD:
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

                add_training_pair(
                    patterns_H[i],
                    zero_offsets,
                    constant_velocities,
                    new_H,
                    new_O,
                    new_V,
                    target_style,
                    target_feel,
                    target_overall_dynamic,
                    tempo[i],
                )

                found_combinations.add((target_style, target_feel))

                # if DEBUG:
                #     reconstruct_midi_patterns(i, j, patterns_H[i], patterns_O[i], patterns_V[i], patterns_H[j], patterns_O[j], patterns_V[j], patterns_H[i], new_O, new_V, tempo[i], instruction)
                continue

            if DEBUG:
                quit()

    
    
    # save to h5py
    with h5py.File("groove_training_pairs_genre.h5", "w") as f:
        source = f.create_group("source")
        source.create_dataset("source_hits", data=np.array(SOURCE_HITS))
        source.create_dataset("source_offsets", data=np.array(SOURCE_OFFSETS))
        source.create_dataset("source_velocities", data=np.array(SOURCE_VELOCITIES))

        target = f.create_group("target")
        target.create_dataset("target_hits", data=np.array(TARGET_HITS))
        target.create_dataset("target_offsets", data=np.array(TARGET_OFFSETS))
        target.create_dataset("target_velocities", data=np.array(TARGET_VELOCITIES))

        metadata = f.create_group("metadata")
        metadata.create_dataset("style", data=np.array(STYLE, dtype='S'))
        metadata.create_dataset("feel", data=np.array(FEEL, dtype='S'))
        metadata.create_dataset("overall_dynamic", data=np.array(DYNAMIC, dtype='S'))
        metadata.create_dataset("dynamic_variation", data=np.array(VARIATION, dtype='S'))
        metadata.create_dataset("tempo", data=np.array(TEMPOS))

if __name__ == "__main__":
    main()