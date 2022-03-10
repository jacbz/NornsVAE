-- thesis (placeholder name)
-- jacob zhang

-- drums or melody
drums = true

engine.name = drums and 'Ack' or 'PolyPerc'
if drums then
  local Ack = require 'ack/lib/ack'
end

util = require 'util'
json = include('lib/json')
MusicUtil = require "musicutil"

server = 'http://192.168.1.23:5000/'

total_steps = 16
bpm = 100
params:set("clock_tempo", bpm)
steps_per_beat = 4
step_length = 1/4 -- in seconds
min_note = 48
max_note = 83
interpolation_steps = 11

initialized = false


-- current sequence
current_step = 1
current_pad_sequence = 'left'

-- attribute vector mode: 1: density, 2: averageInterval
mode = 1
modes = { 
  'DSTY',
  'BD',
  'SD',
  'HH',
  'TO',
  'CY'
}
mode_min = -4
mode_max = 4
mode_steps = mode_max - mode_min
mode_current_step = { 0, 0, 0, 0, 0, 0 }

-- interpolation
current_interpolation = 1
lookahead = {}

-- server communication
job_id = nil
trigger_lookahead_at_step = nil  -- when setting this to a step, a lookahead call will be triggered at that step
trigger_replace_at_step = nil  -- when setting this to a step, a replace call will be triggered at that step

function server_sync()
  local response = util.os_capture("curl -g -s " .. server .. "sync" .. " --max-time 1")
  if string.len(response) > 2 then
    local parsedResponse = json.decode(response)
    if job_id == parsedResponse.job_id then
      lookahead = parsedResponse.data
      job_id = nil
    end
  end
end

function server_lookahead()
  local attr_values = json.encode(mode_current_step)
  local attribute = modes[mode]
  job_id = util.os_capture("curl -g -s '" .. server .. "lookahead?attr_values=" .. attr_values .. "&attribute=" .. attribute .. "' --max-time 1")
end

function server_replace()
  mode_current_step = { 0, 0, 0, 0, 0, 0 }
  local dict1 = json.encode(lookahead[tostring(0)][attr_values_str()])
  local dict2 = json.encode(lookahead[tostring(interpolation_steps-1)][attr_values_str()])
  job_id = util.os_capture("curl -g -s '" .. server .. "replace?dict1=" .. dict1 .. "&dict2=" .. dict2 .. "' --max-time 1")
end

function server_reload()
  mode_current_step = { 0, 0, 0, 0, 0, 0 }
  job_id = util.os_capture("curl -g -s " .. server .. "reload --max-time 1")
end

function get_current_sample()
  return lookahead[tostring(current_interpolation-1)][attr_values_str()]
end

function get_current_notes()
  local dict = get_current_sample()
  if dict then
    return dict.notes
  else
    print("Error: interpolation ".. tostring(current_interpolation-1) .. " not found for " ..  attr_values_str())
    return {}
  end
end

function get_current_pad_notes()
  local interpolation = current_pad_sequence == 'left' and 1 or interpolation_steps
  local dict = lookahead[tostring(interpolation-1)][attr_values_str()]
  if dict then
    return dict.notes
  else
    print("Error: interpolation ".. tostring(interpolation-1) .. " not found for " ..  attr_values_str())
    return {}
  end
end

function attr_values_str()
  local strings = {}
  for i, val in ipairs(mode_current_step) do
    table.insert(strings, modes[i] .. string.format("%+d", val))
  end
  return table.concat(strings, ", ")
end

function init()
  if drums then
    sample_directory = "/home/we/dust/audio/common/808"
    samples = {
      -- "808-CY.wav"
      "808-CP.wav",
      "808-HT.wav",
      -- "808-MT.wav",
      "808-LT.wav",
      "808-OH.wav",
      "808-CH.wav",
      "808-SD.wav",
      "808-BD.wav",
    }
    for i, name in pairs(samples) do
      engine.loadSample(i, sample_directory.."/"..name)
    end
  end

  redraw()
  clock.run(function()
    server_lookahead()
    while initialized == false do
      server_sync()
      if next(lookahead) ~= nil then
        initialized = true
        redraw()
        clock.run(step)
      end
      clock.sleep(1/2)
    end
  end)
end


function step()
  while true do
    clock.sync(step_length)

    if trigger_lookahead_at_step ~= nil and current_step == (trigger_lookahead_at_step % total_steps) then
      trigger_lookahead_at_step = nil
      server_lookahead()
    end

    if trigger_replace_at_step ~= nil and current_step == (trigger_replace_at_step % total_steps) then
      trigger_replace_at_step = nil
      server_replace()
    end

    if job_id ~= nil and current_step % 2 == 0 then
      server_sync()
    end

    local notes = get_current_notes()

    local notes_at_step = notes[tostring(current_step-1)]

    if notes_at_step then
      for i, note in pairs(notes_at_step) do
        if drums then
          engine.trig(note >= 3 and note - 1 or note)
        else
          local pitch = note.pitch
          engine.hz(MusicUtil.note_num_to_freq(pitch))
        end
      end
    end

    current_step = current_step + 1
    if current_step > total_steps then current_step = 1 end
    redraw()
  end
end

function key(n, z)
  -- key actions: n = number, z = state

  if z ~= 1 then return end

  if n == 1 then
    server_reload()
  elseif n == 2 then
    current_pad_sequence = 'left'
  elseif n == 3 then
    current_pad_sequence = 'right'
  end
end

function enc(n, d)
  -- encoder actions: n = number, d = delta
  if n== 1 then
    current_interpolation = util.clamp(current_interpolation + d, 1, interpolation_steps)
  elseif n == 2 then
    mode = util.clamp(mode + d, 1, #modes)
    trigger_lookahead_at_step = current_step + 1
  elseif n == 3 then
    mode_current_step[mode] = util.clamp(mode_current_step[mode] + d, mode_min, mode_max)
  end
  redraw()
end

function draw_sample(sample, offsetX, offsetY)
  if sample ~= nil then
    screen.pixel(sample['x'] + offsetX, sample['y'] + offsetY)
    screen.fill()
  end
end

function redraw()
  screen.clear()

  -- screen.level(4)
  -- screen.move(0, 10)
  -- screen.font_size(8)
  -- screen.text('THESIS')

  if not initialized then
    screen.move(0,20)
    screen.level(15)
    screen.text('Connecting to server...')
    screen.update()
    return
  end

  if job_id ~= nil then
    screen.level(4)
    screen.move(0, 10)
    screen.font_size(8)
    screen.text('LOADING')
  end

  local notes = get_current_notes()
  for step = 1, total_steps do
    local notes_at_step = notes[tostring(step-1)]
    if notes_at_step then      
      for i, note in pairs(notes_at_step) do
        pitch = drums and note or note.pitch
        duration = drums and 1 or note.duration
        
        level = (current_step >= step and current_step < step + duration) and 15 or 4
        screen.level(level)
        if drums then
          screen.rect((step - 1) *  4 + 6, 6 + pitch * 5, 3, 3)
          screen.fill()
        else
          screen.move((step - 1) * 2 + 4, 16 - pitch + max_note)
          screen.line_rel(2 * duration, 0)
          screen.stroke()
        end
      end
    end
  end

  if drums then
    g:all(0)
    local pad_notes = get_current_pad_notes()
    for step = 1, total_steps do
      local notes_at_step = pad_notes[tostring(step-1)]
      if notes_at_step then      
        for i, note in pairs(notes_at_step) do
          level = (current_step == step) and 15 or 4
          screen.level(level)
          if drums then
            if step < 17 then
              g:led(step, note, level)
            end
          end
        end
      end
      g:refresh()
    end
  end

  -- map
  local left = 128 - 50
  local top = 1
  screen.level(8)
  screen.rect(left, top, 50, 50)
  screen.stroke()

  for step = 1, interpolation_steps do
    screen.level(step == current_interpolation and 15 or 4)
    draw_sample(lookahead[tostring(step-1)][attr_values_str()], left, top)  
  end  

  -- current step
  -- screen.level(1)
  -- screen.rect((current_step - 1) * 4, 0, 1, 64)
  -- screen.fill()

  -- divider line
  screen.level(2)
  screen.move(0, 55)
  screen.line_rel(128, 0)
  screen.stroke()

  -- interpolation progress bar
  screen.level(15)
  screen.move((current_interpolation - 1) * (128 / interpolation_steps), 55)
  screen.line_rel(128 / interpolation_steps, 0)
  screen.stroke()
  
  -- attribute vector modes
  screen.move(0, 62)
  for m = 1, #modes do
    local mode_name = modes[m]

    screen.level(mode == m and 15 or 4)
    screen.text(mode_name .. string.format("%+d", mode_current_step[m]))
    screen.move_rel(4, 0)
  end

  screen.update()
end

local grid = util.file_exists(_path.code.."midigrid") and include "midigrid/lib/mg_128" or grid
g = grid.connect()

g.key = function (x, y, z)
  if (z ~= 1) then return end
  toggleDrum(x, y)
end

function toggleDrum(x, y)
  local notes = get_current_pad_notes()
  if notes[tostring(x-1)] == nil then
    notes[tostring(x-1)] = {}
  end

  get_current_sample().hash = nil

  -- if note already exists, toggle off
  for i, note in pairs(notes[tostring(x-1)]) do
    if note == y then
      table.remove(notes[tostring(x-1)], i)
      trigger_replace_at_step = current_step + 4
      return
    end
  end

  -- toggle on
  table.insert(notes[tostring(x-1)], y)
  trigger_replace_at_step = current_step + 4
end