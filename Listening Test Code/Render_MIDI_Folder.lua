-- Import MIDI onto the selected Addictive Drums track,
-- detect and round the MIDI tempo,
-- change the project tempo,
-- play the MIDI once, and render to MP3.

local SCRIPT_NAME = "Render MIDI with Addictive Drums"
local project = 0

-- Safety limit for very large batches. Set to 0 for unlimited.
-- Because this system previously crashed around 1000 renders,
-- 500 provides a conservative checkpoint. Run the script again
-- to continue; existing MP3s are skipped.
local MAX_RENDERS_PER_RUN = 800

-- A normal synchronous ReaScript can create one implicit
-- "ReaScript: Run" undo state. Scheduling an empty deferred
-- callback suppresses that implicit undo point.
reaper.defer(function() end)

------------------------------------------------------------
-- Standard MIDI file tempo detection
------------------------------------------------------------

local function read_u16_be(data, position)
    local byte_1, byte_2 = data:byte(position, position + 1)

    if not byte_1 or not byte_2 then
        return nil
    end

    return (byte_1 * 256) + byte_2
end

local function read_u32_be(data, position)
    local byte_1, byte_2, byte_3, byte_4 =
        data:byte(position, position + 3)

    if not byte_1 or not byte_2 or
       not byte_3 or not byte_4 then
        return nil
    end

    return
        (byte_1 * 16777216) +
        (byte_2 * 65536) +
        (byte_3 * 256) +
        byte_4
end

-- Read a MIDI variable-length quantity.
local function read_vlq(data, position, limit)
    local value = 0

    -- Standard MIDI VLQs use no more than four bytes.
    for _ = 1, 4 do
        if position > limit then
            return nil, position,
                "Unexpected end of file while reading a MIDI value."
        end

        local byte = data:byte(position)
        position = position + 1

        value = (value * 128) + (byte % 128)

        if byte < 128 then
            return value, position, nil
        end
    end

    return nil, position,
        "Invalid variable-length quantity in the MIDI file."
end

local function ensure_bytes_available(
    position,
    byte_count,
    limit
)
    if byte_count < 0 then
        return false
    end

    return position + byte_count - 1 <= limit
end

local function parse_midi_track_for_tempo(
    data,
    track_start,
    track_end
)
    local position = track_start
    local absolute_tick = 0
    local running_status = nil

    local earliest_bpm = nil
    local earliest_tick = math.huge

    while position <= track_end do
        ----------------------------------------------------
        -- Delta time
        ----------------------------------------------------

        local delta_time
        local delta_error

        delta_time, position, delta_error =
            read_vlq(data, position, track_end)

        if not delta_time then
            return nil, nil, delta_error
        end

        absolute_tick = absolute_tick + delta_time

        if position > track_end then
            return nil, nil,
                "The MIDI track ended before an event was found."
        end

        ----------------------------------------------------
        -- Status byte or running status
        ----------------------------------------------------

        local first_byte = data:byte(position)
        local status

        if first_byte >= 0x80 then
            status = first_byte
            position = position + 1

            if status < 0xF0 then
                running_status = status
            else
                -- System, SysEx and meta events cancel
                -- channel-message running status.
                running_status = nil
            end
        else
            status = running_status

            if not status then
                return nil, nil,
                    "Invalid MIDI running-status event."
            end
        end

        ----------------------------------------------------
        -- Channel MIDI messages
        ----------------------------------------------------

        if status < 0xF0 then
            local message_type = status - (status % 16)

            local data_length

            if message_type == 0xC0 or
               message_type == 0xD0 then
                data_length = 1
            else
                data_length = 2
            end

            if not ensure_bytes_available(
                position,
                data_length,
                track_end
            ) then
                return nil, nil,
                    "A MIDI channel event is incomplete."
            end

            position = position + data_length

        ----------------------------------------------------
        -- Meta event
        ----------------------------------------------------

        elseif status == 0xFF then
            if position > track_end then
                return nil, nil,
                    "A MIDI meta-event is incomplete."
            end

            local meta_type = data:byte(position)
            position = position + 1

            local meta_length
            local length_error

            meta_length, position, length_error =
                read_vlq(data, position, track_end)

            if not meta_length then
                return nil, nil, length_error
            end

            if not ensure_bytes_available(
                position,
                meta_length,
                track_end
            ) then
                return nil, nil,
                    "A MIDI meta-event exceeds the track length."
            end

            -- Set Tempo:
            -- FF 51 03 tt tt tt
            --
            -- The three data bytes contain microseconds
            -- per quarter note as a 24-bit big-endian value.
            if meta_type == 0x51 and meta_length == 3 then
                local tempo_byte_1,
                      tempo_byte_2,
                      tempo_byte_3 =
                    data:byte(position, position + 2)

                local microseconds_per_quarter =
                    (tempo_byte_1 * 65536) +
                    (tempo_byte_2 * 256) +
                    tempo_byte_3

                if microseconds_per_quarter > 0 and
                   absolute_tick < earliest_tick then

                    earliest_bpm =
                        60000000 /
                        microseconds_per_quarter

                    earliest_tick = absolute_tick
                end
            end

            position = position + meta_length

            -- End of Track.
            if meta_type == 0x2F then
                break
            end

        ----------------------------------------------------
        -- System Exclusive events
        ----------------------------------------------------

        elseif status == 0xF0 or status == 0xF7 then
            local sysex_length
            local length_error

            sysex_length, position, length_error =
                read_vlq(data, position, track_end)

            if not sysex_length then
                return nil, nil, length_error
            end

            if not ensure_bytes_available(
                position,
                sysex_length,
                track_end
            ) then
                return nil, nil,
                    "A MIDI SysEx event exceeds the track length."
            end

            position = position + sysex_length

        ----------------------------------------------------
        -- Other system messages
        ----------------------------------------------------

        else
            local system_data_lengths = {
                [0xF1] = 1,
                [0xF2] = 2,
                [0xF3] = 1,
                [0xF4] = 0,
                [0xF5] = 0,
                [0xF6] = 0,
                [0xF8] = 0,
                [0xF9] = 0,
                [0xFA] = 0,
                [0xFB] = 0,
                [0xFC] = 0,
                [0xFD] = 0,
                [0xFE] = 0
            }

            local data_length =
                system_data_lengths[status]

            if data_length == nil then
                return nil, nil,
                    string.format(
                        "Unsupported MIDI status byte: 0x%02X",
                        status
                    )
            end

            if not ensure_bytes_available(
                position,
                data_length,
                track_end
            ) then
                return nil, nil,
                    "A MIDI system event is incomplete."
            end

            position = position + data_length
        end
    end

    return earliest_bpm, earliest_tick, nil
end

local function detect_midi_tempo(midi_path)
    local midi_file, open_error =
        io.open(midi_path, "rb")

    if not midi_file then
        return nil,
            "Could not open the MIDI file:\n\n" ..
            tostring(open_error)
    end

    local data = midi_file:read("*a")
    midi_file:close()

    if not data or #data < 14 then
        return nil,
            "The selected file is too short to be a valid MIDI file."
    end

    if data:sub(1, 4) ~= "MThd" then
        return nil,
            "The selected file is not a standard MIDI file."
    end

    local header_length = read_u32_be(data, 5)

    if not header_length or header_length < 6 then
        return nil,
            "The MIDI file contains an invalid header."
    end

    if 8 + header_length > #data then
        return nil,
            "The MIDI header exceeds the file length."
    end

    local track_count = read_u16_be(data, 11)

    if not track_count or track_count < 1 then
        return nil,
            "The MIDI file does not contain any tracks."
    end

    local position = 9 + header_length
    local tracks_read = 0

    local earliest_bpm = nil
    local earliest_tick = math.huge

    -- Read chunks until all declared MIDI tracks have been found.
    while position + 7 <= #data and
          tracks_read < track_count do

        local chunk_id =
            data:sub(position, position + 3)

        local chunk_length =
            read_u32_be(data, position + 4)

        if not chunk_length then
            return nil,
                "A MIDI chunk contains an invalid length."
        end

        local chunk_start = position + 8
        local chunk_end =
            chunk_start + chunk_length - 1

        if chunk_end > #data then
            return nil,
                "A MIDI chunk exceeds the file length."
        end

        if chunk_id == "MTrk" then
            tracks_read = tracks_read + 1

            local track_bpm
            local track_tick
            local track_error

            track_bpm, track_tick, track_error =
                parse_midi_track_for_tempo(
                    data,
                    chunk_start,
                    chunk_end
                )

            if track_error then
                return nil,
                    "Could not parse MIDI track " ..
                    tostring(tracks_read) ..
                    ":\n\n" ..
                    track_error
            end

            if track_bpm and
               track_tick < earliest_tick then

                earliest_bpm = track_bpm
                earliest_tick = track_tick
            end
        end

        position = chunk_end + 1
    end

    if tracks_read < track_count then
        return nil,
            "The MIDI file ended before all declared tracks were found."
    end

    if not earliest_bpm then
        return nil,
            "No Set Tempo event was found in the MIDI file."
    end

    return earliest_bpm, nil
end

------------------------------------------------------------
-- Require one selected Addictive Drums track
------------------------------------------------------------

------------------------------------------------------------
-- Path and folder helpers
------------------------------------------------------------

local path_separator = package.config:sub(1, 1)

local function join_path(directory, filename)
    if directory:sub(-1) == "/" or
       directory:sub(-1) == "\\" then
        return directory .. filename
    end

    return directory .. path_separator .. filename
end

local function ensure_trailing_separator(directory)
    if directory:sub(-1) == "/" or
       directory:sub(-1) == "\\" then
        return directory
    end

    return directory .. path_separator
end

local function get_filename_without_extension(filename)
    return filename:gsub("%.[^%.]+$", "")
end

local function is_midi_filename(filename)
    local lower_filename = filename:lower()

    return lower_filename:match("%.mid$") ~= nil or
           lower_filename:match("%.midi$") ~= nil
end

local function enumerate_midi_files(directory)
    local midi_files = {}

    -- Invalidate REAPER's cached directory listing.
    reaper.EnumerateFiles(directory, -1)

    local index = 0

    while true do
        local filename =
            reaper.EnumerateFiles(directory, index)

        if not filename then
            break
        end

        if is_midi_filename(filename) then
            midi_files[#midi_files + 1] = filename
        end

        index = index + 1
    end

    table.sort(
        midi_files,
        function(first, second)
            return first:lower() < second:lower()
        end
    )

    return midi_files
end

------------------------------------------------------------
-- Track-selection helpers
------------------------------------------------------------

local function capture_selected_tracks()
    local selected_tracks = {}

    for i = 0, reaper.CountSelectedTracks(project) - 1 do
        selected_tracks[#selected_tracks + 1] =
            reaper.GetSelectedTrack(project, i)
    end

    return selected_tracks
end

local function restore_selected_tracks(selected_tracks)
    for i = 0, reaper.CountTracks(project) - 1 do
        local track = reaper.GetTrack(project, i)
        reaper.SetTrackSelected(track, false)
    end

    for _, track in ipairs(selected_tracks) do
        if reaper.ValidatePtr2(
            project,
            track,
            "MediaTrack*"
        ) then
            reaper.SetTrackSelected(track, true)
        end
    end
end

------------------------------------------------------------
-- Tempo-map preservation
------------------------------------------------------------

local function delete_all_tempo_markers()
    for i =
        reaper.CountTempoTimeSigMarkers(project) - 1,
        0,
        -1
    do
        reaper.DeleteTempoTimeSigMarker(project, i)
    end
end

local function capture_tempo_state()
    local previous_cursor = reaper.GetCursorPosition()

    -- Read the underlying project tempo at the beginning.
    reaper.SetEditCurPos(0, false, false)

    local tempo_state = {
        base_bpm = reaper.Master_GetTempo(),
        markers = {}
    }

    for i = 0,
        reaper.CountTempoTimeSigMarkers(project) - 1
    do
        local success,
              time_position,
              measure_position,
              beat_position,
              bpm,
              numerator,
              denominator,
              linear_tempo =
            reaper.GetTempoTimeSigMarker(project, i)

        if success then
            tempo_state.markers[
                #tempo_state.markers + 1
            ] = {
                time_position = time_position,
                bpm = bpm,
                numerator = numerator,
                denominator = denominator,
                linear_tempo = linear_tempo
            }
        end
    end

    reaper.SetEditCurPos(
        previous_cursor,
        false,
        false
    )

    return tempo_state
end

local function restore_tempo_state(
    tempo_state,
    cursor_position
)
    delete_all_tempo_markers()

    reaper.SetEditCurPos(0, false, false)

    reaper.SetCurrentBPM(
        project,
        tempo_state.base_bpm,
        false
    )

    for _, marker in ipairs(tempo_state.markers) do
        reaper.SetTempoTimeSigMarker(
            project,
            -1,
            marker.time_position,
            -1,
            -1,
            marker.bpm,
            marker.numerator,
            marker.denominator,
            marker.linear_tempo
        )
    end

    reaper.SetEditCurPos(
        cursor_position,
        false,
        false
    )

    reaper.UpdateTimeline()
end

local function set_constant_project_tempo(
    bpm,
    cursor_position
)
    -- During the batch, the original tempo map stays removed.
    -- Only delete markers if the MIDI importer added new ones.
    if reaper.CountTempoTimeSigMarkers(project) > 0 then
        delete_all_tempo_markers()
    end

    reaper.SetEditCurPos(0, false, false)

    reaper.SetCurrentBPM(
        project,
        bpm,
        false
    )

    reaper.SetEditCurPos(
        cursor_position,
        false,
        false
    )

    reaper.UpdateTimeline()
end

------------------------------------------------------------
-- Project item and track snapshots
------------------------------------------------------------

local function capture_existing_items()
    local existing_items = {}

    for i = 0, reaper.CountMediaItems(project) - 1 do
        local item = reaper.GetMediaItem(project, i)
        existing_items[item] = true
    end

    return existing_items
end

local function capture_existing_tracks()
    local existing_tracks = {}

    for i = 0, reaper.CountTracks(project) - 1 do
        local track = reaper.GetTrack(project, i)
        existing_tracks[track] = true
    end

    return existing_tracks
end

local function find_imported_content(
    existing_items,
    existing_tracks
)
    local imported_items = {}
    local new_tracks = {}
    local new_tracks_lookup = {}

    for i = 0, reaper.CountTracks(project) - 1 do
        local track = reaper.GetTrack(project, i)

        if not existing_tracks[track] then
            new_tracks[#new_tracks + 1] = track
            new_tracks_lookup[track] = true
        end
    end

    for i = 0, reaper.CountMediaItems(project) - 1 do
        local item = reaper.GetMediaItem(project, i)

        if not existing_items[item] then
            imported_items[#imported_items + 1] = item
        end
    end

    return imported_items,
           new_tracks,
           new_tracks_lookup
end

local function remove_imported_content(
    imported_items,
    new_tracks,
    new_tracks_lookup
)
    -- First remove imported items placed on existing tracks.
    for i = #imported_items, 1, -1 do
        local item = imported_items[i]

        if reaper.ValidatePtr2(
            project,
            item,
            "MediaItem*"
        ) then
            local track =
                reaper.GetMediaItem_Track(item)

            if not new_tracks_lookup[track] then
                reaper.DeleteTrackMediaItem(
                    track,
                    item
                )
            end
        end
    end

    -- Then remove any tracks created by REAPER's MIDI importer.
    for i = #new_tracks, 1, -1 do
        local track = new_tracks[i]

        if reaper.ValidatePtr2(
            project,
            track,
            "MediaTrack*"
        ) then
            reaper.DeleteTrack(track)
        end
    end
end

------------------------------------------------------------
-- Render settings
------------------------------------------------------------

local numeric_setting_names = {
    "RENDER_SETTINGS",
    "RENDER_BOUNDSFLAG",
    "RENDER_STARTPOS",
    "RENDER_ENDPOS",
    "RENDER_CHANNELS",
    "RENDER_TAILFLAG",
    "RENDER_TAILMS",
    "RENDER_ADDTOPROJ"
}

local string_setting_names = {
    "RENDER_FILE",
    "RENDER_PATTERN",
    "RENDER_FORMAT",
    "RENDER_FORMAT2"
}

local function capture_render_settings()
    local settings = {
        numeric = {},
        strings = {}
    }

    for _, name in ipairs(numeric_setting_names) do
        settings.numeric[name] =
            reaper.GetSetProjectInfo(
                project,
                name,
                0,
                false
            )
    end

    for _, name in ipairs(string_setting_names) do
        local _, value =
            reaper.GetSetProjectInfo_String(
                project,
                name,
                "",
                false
            )

        settings.strings[name] = value
    end

    return settings
end

local function restore_render_settings(settings)
    for _, name in ipairs(numeric_setting_names) do
        reaper.GetSetProjectInfo(
            project,
            name,
            settings.numeric[name],
            true
        )
    end

    for _, name in ipairs(string_setting_names) do
        reaper.GetSetProjectInfo_String(
            project,
            name,
            settings.strings[name],
            true
        )
    end
end

local function configure_mp3_render(
    output_directory,
    render_filename,
    render_start,
    render_end
)
    -- Render the master mix.
    reaper.GetSetProjectInfo(
        project,
        "RENDER_SETTINGS",
        0,
        true
    )

    -- Use custom time bounds.
    reaper.GetSetProjectInfo(
        project,
        "RENDER_BOUNDSFLAG",
        0,
        true
    )

    reaper.GetSetProjectInfo(
        project,
        "RENDER_STARTPOS",
        render_start,
        true
    )

    reaper.GetSetProjectInfo(
        project,
        "RENDER_ENDPOS",
        render_end,
        true
    )

    -- Stereo.
    reaper.GetSetProjectInfo(
        project,
        "RENDER_CHANNELS",
        2,
        true
    )

    -- No additional tail.
    reaper.GetSetProjectInfo(
        project,
        "RENDER_TAILFLAG",
        0,
        true
    )

    reaper.GetSetProjectInfo(
        project,
        "RENDER_ADDTOPROJ",
        0,
        true
    )

    reaper.GetSetProjectInfo_String(
        project,
        "RENDER_FILE",
        ensure_trailing_separator(output_directory),
        true
    )

    reaper.GetSetProjectInfo_String(
        project,
        "RENDER_PATTERN",
        render_filename,
        true
    )

    -- MP3 using REAPER's current/default MP3 settings.
    reaper.GetSetProjectInfo_String(
        project,
        "RENDER_FORMAT",
        "l3pm",
        true
    )

    reaper.GetSetProjectInfo_String(
        project,
        "RENDER_FORMAT2",
        "",
        true
    )
end


------------------------------------------------------------
-- Deterministic output-name handling
------------------------------------------------------------

local function sanitize_render_base(midi_filename)
    local base_name =
        get_filename_without_extension(midi_filename)

    -- A dollar sign starts a REAPER render wildcard.
    base_name = base_name:gsub("%$", "_")

    if base_name == "" then
        base_name = "render"
    end

    return base_name
end

local function build_render_filename_map(midi_files)
    local base_counts = {}
    local filename_map = {}
    local used_names = {}

    -- Count duplicate base names first.
    for _, midi_filename in ipairs(midi_files) do
        local base_name =
            sanitize_render_base(midi_filename)

        local key = base_name:lower()

        base_counts[key] =
            (base_counts[key] or 0) + 1
    end

    -- Generate stable names that do not depend on which
    -- output files already exist.
    for _, midi_filename in ipairs(midi_files) do
        local base_name =
            sanitize_render_base(midi_filename)

        local candidate = base_name

        -- Distinguish "name.mid" from "name.midi".
        if base_counts[base_name:lower()] > 1 then
            local extension =
                midi_filename:match("%.([^%.]+)$")
                or "midi"

            candidate =
                base_name ..
                "_" ..
                extension:lower()
        end

        local original_candidate = candidate
        local suffix = 2

        while used_names[candidate:lower()] do
            candidate =
                original_candidate ..
                "_" ..
                tostring(suffix)

            suffix = suffix + 1
        end

        used_names[candidate:lower()] = true
        filename_map[midi_filename] = candidate
    end

    return filename_map
end

------------------------------------------------------------
-- Reusable file-backed MIDI item
------------------------------------------------------------

local function create_reusable_midi_item(
    track,
    start_position
)
    local item =
        reaper.AddMediaItemToTrack(track)

    if not item then
        return nil, nil,
            "Could not create the temporary MIDI item."
    end

    local take =
        reaper.AddTakeToMediaItem(item)

    if not take then
        reaper.DeleteTrackMediaItem(track, item)

        return nil, nil,
            "Could not create the temporary MIDI take."
    end

    local empty_source =
        reaper.PCM_Source_CreateFromType("MIDI")

    if not empty_source then
        reaper.DeleteTrackMediaItem(track, item)

        return nil, nil,
            "Could not create an empty MIDI source."
    end

    if not reaper.SetMediaItemTake_Source(
        take,
        empty_source
    ) then
        reaper.PCM_Source_Destroy(empty_source)
        reaper.DeleteTrackMediaItem(track, item)

        return nil, nil,
            "Could not attach the empty MIDI source."
    end

    reaper.SetMediaItemInfo_Value(
        item,
        "D_POSITION",
        start_position
    )

    reaper.SetMediaItemInfo_Value(
        item,
        "D_LENGTH",
        0.001
    )

    reaper.SetMediaItemInfo_Value(
        item,
        "B_LOOPSRC",
        0
    )

    -- Beats: position, length and rate.
    reaper.SetMediaItemInfo_Value(
        item,
        "C_BEATATTACHMODE",
        1
    )

    return item, take, nil
end

local function replace_take_source(
    take,
    new_source
)
    local old_source =
        reaper.GetMediaItemTake_Source(take)

    if not reaper.SetMediaItemTake_Source(
        take,
        new_source
    ) then
        return false,
            "Could not replace the temporary MIDI source."
    end

    -- SetMediaItemTake_Source does not destroy the old source.
    if old_source then
        reaper.PCM_Source_Destroy(old_source)
    end

    return true, nil
end

local function clear_reusable_midi_take(
    item,
    take
)
    local empty_source =
        reaper.PCM_Source_CreateFromType("MIDI")

    if not empty_source then
        return false,
            "Could not create a replacement empty MIDI source."
    end

    local replaced, replace_error =
        replace_take_source(
            take,
            empty_source
        )

    if not replaced then
        reaper.PCM_Source_Destroy(empty_source)
        return false, replace_error
    end

    reaper.SetMediaItemInfo_Value(
        item,
        "D_LENGTH",
        0.001
    )

    return true, nil
end

local function load_midi_into_reusable_item(
    item,
    take,
    midi_path,
    start_position
)
    -- true keeps MIDI file-backed instead of converting it
    -- into an in-project MIDI event source.
    local source =
        reaper.PCM_Source_CreateFromFileEx(
            midi_path,
            true
        )

    if not source then
        return nil,
            "Could not create a file-backed MIDI source."
    end

    local source_length, length_is_qn =
        reaper.GetMediaSourceLength(source)

    if not source_length or
       source_length <= 0 then

        reaper.PCM_Source_Destroy(source)

        return nil,
            "The MIDI source has no usable length."
    end

    local replaced, replace_error =
        replace_take_source(
            take,
            source
        )

    if not replaced then
        reaper.PCM_Source_Destroy(source)
        return nil, replace_error
    end

    local render_end

    if length_is_qn then
        local start_qn =
            reaper.TimeMap2_timeToQN(
                project,
                start_position
            )

        render_end =
            reaper.TimeMap2_QNToTime(
                project,
                start_qn + source_length
            )
    else
        render_end =
            start_position + source_length
    end

    if not render_end or
       render_end <= start_position then

        clear_reusable_midi_take(item, take)

        return nil,
            "Could not calculate the MIDI render length."
    end

    reaper.SetMediaItemInfo_Value(
        item,
        "D_POSITION",
        start_position
    )

    reaper.SetMediaItemInfo_Value(
        item,
        "D_LENGTH",
        render_end - start_position
    )

    reaper.SetMediaItemInfo_Value(
        item,
        "B_LOOPSRC",
        0
    )

    reaper.SetMediaItemTakeInfo_Value(
        take,
        "D_STARTOFFS",
        0
    )

    reaper.SetMediaItemTakeInfo_Value(
        take,
        "D_PLAYRATE",
        1
    )

    return render_end, nil
end

------------------------------------------------------------
-- Render one MIDI file
------------------------------------------------------------

local function render_midi_file(
    reusable_item,
    reusable_take,
    midi_directory,
    midi_filename,
    output_directory,
    render_filename,
    import_position
)
    local midi_path =
        join_path(midi_directory, midi_filename)

    --------------------------------------------------------
    -- Detect and apply the file tempo
    --------------------------------------------------------

    local detected_bpm, tempo_error =
        detect_midi_tempo(midi_path)

    if not detected_bpm then
        return false,
            "Tempo detection failed: " ..
            tostring(tempo_error)
    end

    local rounded_bpm =
        math.floor(detected_bpm + 0.5)

    if rounded_bpm <= 0 then
        return false,
            "The detected MIDI tempo was invalid."
    end

    set_constant_project_tempo(
        rounded_bpm,
        import_position
    )

    --------------------------------------------------------
    -- Attach this file to the one reusable item
    --------------------------------------------------------

    local render_end, load_error =
        load_midi_into_reusable_item(
            reusable_item,
            reusable_take,
            midi_path,
            import_position
        )

    if not render_end then
        return false, load_error
    end

    reaper.UpdateArrange()

    --------------------------------------------------------
    -- Configure and run render
    --------------------------------------------------------

    configure_mp3_render(
        output_directory,
        render_filename,
        import_position,
        render_end
    )

    local output_path =
        join_path(
            output_directory,
            render_filename .. ".mp3"
        )

    -- File: Render project, using the most recent settings.
    reaper.Main_OnCommand(41824, 0)

    local rendered_successfully =
        reaper.file_exists(output_path)

    --------------------------------------------------------
    -- Explicitly release this file's MIDI source
    --------------------------------------------------------

    local cleared, clear_error =
        clear_reusable_midi_take(
            reusable_item,
            reusable_take
        )

    reaper.UpdateArrange()

    if not cleared then
        return false,
            "The MP3 render completed, but the MIDI source " ..
            "could not be released: " ..
            tostring(clear_error)
    end

    if not rendered_successfully then
        return false,
            "The render completed, but the expected " ..
            "MP3 file was not found."
    end

    return true, {
        output_path = output_path,
        detected_bpm = detected_bpm,
        rounded_bpm = rounded_bpm
    }
end

------------------------------------------------------------
-- Require one selected Addictive Drums track
------------------------------------------------------------

local drums_track =
    reaper.GetSelectedTrack(project, 0)

if not drums_track then
    reaper.ShowMessageBox(
        "Select your Addictive Drums track before " ..
        "running this script.",
        SCRIPT_NAME,
        0
    )

    return
end

------------------------------------------------------------
-- Choose source and output folders
------------------------------------------------------------

local source_ok, midi_directory =
    reaper.GetUserFileName(
        3,
        "Choose the folder containing MIDI files",
        reaper.GetProjectPath(""),
        ""
    )

if not source_ok then
    return
end

local midi_files =
    enumerate_midi_files(midi_directory)

if #midi_files == 0 then
    reaper.ShowMessageBox(
        "No .mid or .midi files were found in:\n\n" ..
        midi_directory,
        SCRIPT_NAME,
        0
    )

    return
end

local output_ok, output_directory =
    reaper.GetUserFileName(
        3,
        "Choose the MP3 output folder",
        midi_directory,
        ""
    )

if not output_ok then
    return
end

------------------------------------------------------------
-- Preserve project state
------------------------------------------------------------

local original_cursor =
    reaper.GetCursorPosition()

local original_selected_tracks =
    capture_selected_tracks()

local original_tempo_state =
    capture_tempo_state()

local original_render_settings =
    capture_render_settings()

local render_filename_map =
    build_render_filename_map(midi_files)

local successful_count = 0
local skipped_count = 0
local failed_count = 0
local examined_count = 0
local stopped_at_safety_limit = false

local failed_renders = {}
local maximum_stored_failures = 15

------------------------------------------------------------
-- Create one reusable MIDI item for the whole run
------------------------------------------------------------

set_constant_project_tempo(
    original_tempo_state.base_bpm,
    original_cursor
)

reaper.SetOnlyTrackSelected(drums_track)

local reusable_item,
      reusable_take,
      reusable_error =
    create_reusable_midi_item(
        drums_track,
        original_cursor
    )

if not reusable_item then
    restore_render_settings(
        original_render_settings
    )

    restore_tempo_state(
        original_tempo_state,
        original_cursor
    )

    restore_selected_tracks(
        original_selected_tracks
    )

    reaper.ShowMessageBox(
        reusable_error,
        SCRIPT_NAME,
        0
    )

    return
end

------------------------------------------------------------
-- Batch render
------------------------------------------------------------

reaper.ClearConsole()

reaper.ShowConsoleMsg(
    string.format(
        "%s\n\n" ..
        "Source folder: %s\n" ..
        "Output folder: %s\n" ..
        "MIDI files found: %d\n" ..
        "Maximum new renders this run: %s\n\n",
        SCRIPT_NAME,
        midi_directory,
        output_directory,
        #midi_files,
        MAX_RENDERS_PER_RUN == 0
            and "unlimited"
            or tostring(MAX_RENDERS_PER_RUN)
    )
)

local batch_ok, batch_error =
    xpcall(
        function()
            for index, midi_filename in ipairs(midi_files) do
                examined_count = index

                local render_filename =
                    render_filename_map[midi_filename]

                local expected_output_path =
                    join_path(
                        output_directory,
                        render_filename .. ".mp3"
                    )

                ------------------------------------------------
                -- Resume support
                ------------------------------------------------

                if reaper.file_exists(expected_output_path) then
                    skipped_count = skipped_count + 1
                else
                    if MAX_RENDERS_PER_RUN > 0 and
                       successful_count + failed_count >=
                           MAX_RENDERS_PER_RUN then

                        stopped_at_safety_limit = true
                        break
                    end

                    local success, result =
                        render_midi_file(
                            reusable_item,
                            reusable_take,
                            midi_directory,
                            midi_filename,
                            output_directory,
                            render_filename,
                            original_cursor
                        )

                    if success then
                        successful_count =
                            successful_count + 1
                    else
                        failed_count =
                            failed_count + 1

                        if #failed_renders <
                           maximum_stored_failures then

                            failed_renders[
                                #failed_renders + 1
                            ] = {
                                midi_filename =
                                    midi_filename,

                                error_message =
                                    tostring(result)
                            }
                        end

                        reaper.ShowConsoleMsg(
                            "Failed: " ..
                            midi_filename ..
                            "\n" ..
                            tostring(result) ..
                            "\n\n"
                        )
                    end
                end

                ------------------------------------------------
                -- Low-volume progress and Lua cleanup
                ------------------------------------------------

                if index % 100 == 0 then
                    collectgarbage("collect")

                    reaper.ShowConsoleMsg(
                        string.format(
                            "Progress: %d/%d | " ..
                            "Rendered: %d | " ..
                            "Skipped: %d | " ..
                            "Failed: %d\n",
                            index,
                            #midi_files,
                            successful_count,
                            skipped_count,
                            failed_count
                        )
                    )
                else
                    collectgarbage("step", 200)
                end
            end
        end,
        function(error_message)
            if debug and debug.traceback then
                return debug.traceback(
                    tostring(error_message),
                    2
                )
            end

            return tostring(error_message)
        end
    )

------------------------------------------------------------
-- Remove the reusable item and restore state once
------------------------------------------------------------

if reaper.ValidatePtr2(
    project,
    reusable_item,
    "MediaItem*"
) then
    reaper.DeleteTrackMediaItem(
        drums_track,
        reusable_item
    )
end

restore_render_settings(
    original_render_settings
)

restore_tempo_state(
    original_tempo_state,
    original_cursor
)

restore_selected_tracks(
    original_selected_tracks
)

reaper.SetEditCurPos(
    original_cursor,
    false,
    false
)

reaper.UpdateTimeline()
reaper.UpdateArrange()

collectgarbage("collect")

------------------------------------------------------------
-- Report unexpected script error
------------------------------------------------------------

if not batch_ok then
    reaper.ShowConsoleMsg(
        "\nBatch stopped because of an unexpected error:\n\n" ..
        tostring(batch_error) ..
        "\n"
    )

    reaper.ShowMessageBox(
        string.format(
            "The batch stopped because of an unexpected error.\n\n" ..
            "Rendered during this run: %d\n" ..
            "Existing MP3s skipped: %d\n" ..
            "File failures: %d\n\n" ..
            "The original tempo map and render settings " ..
            "were restored.\n\n" ..
            "Run the script again to resume. Existing MP3s " ..
            "will be skipped.\n\n" ..
            "See the REAPER console for details.",
            successful_count,
            skipped_count,
            failed_count
        ),
        SCRIPT_NAME,
        0
    )

    return
end

------------------------------------------------------------
-- Report result
------------------------------------------------------------

local report

if stopped_at_safety_limit then
    report = string.format(
        "Checkpoint reached safely.\n\n" ..
        "Rendered during this run: %d\n" ..
        "Existing MP3s skipped: %d\n" ..
        "Failed: %d\n\n" ..
        "Run the script again to continue. Existing MP3s " ..
        "will be skipped automatically.\n\n" ..
        "Output folder:\n%s",
        successful_count,
        skipped_count,
        failed_count,
        output_directory
    )
else
    report = string.format(
        "Batch render complete.\n\n" ..
        "MIDI files found: %d\n" ..
        "Rendered during this run: %d\n" ..
        "Existing MP3s skipped: %d\n" ..
        "Failed: %d\n\n" ..
        "Output folder:\n%s",
        #midi_files,
        successful_count,
        skipped_count,
        failed_count,
        output_directory
    )
end

if failed_count > 0 then
    report = report .. "\n\nFailed files:"

    for _, failure in ipairs(failed_renders) do
        report = report ..
            "\n\n" ..
            failure.midi_filename ..
            "\n" ..
            failure.error_message
    end

    if failed_count > #failed_renders then
        report = report ..
            string.format(
                "\n\n…and %d more. See the " ..
                "REAPER console for details.",
                failed_count - #failed_renders
            )
    end
end

reaper.ShowMessageBox(
    report,
    SCRIPT_NAME,
    0
)
