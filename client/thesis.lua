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

total_steps = 32
bpm = 90
steps_per_beat = 4
-- step_length = 60 / (bpm * steps_per_beat) -- in seconds
step_length = 1 / 6 -- in seconds
min_note = 48
max_note = 83
interpolation_steps = 11

initialized = false


-- current sequence
current_step = 1

-- attribute vector mode: 1: density, 2: averageInterval
mode = 1
modes = { "DSTY" }
mode_min = -4
mode_max = 4
mode_steps = mode_max - mode_min
mode_current_step = { 0 }

-- interpolation
current_interpolation = 1

lookahead = {}

-- server communication
job_id = nil

function server_sync()
  local response = util.os_capture("curl -g -s " .. server .. "sync")
  if string.len(response) > 2 then
    local parsedResponse = json.parse(response)
    if job_id == parsedResponse.job_id then
      lookahead = parsedResponse.data
      job_id = nil
    end
  end
end

function server_lookahead()
  local attr_values = json.stringify(mode_current_step)
  local attribute = modes[mode]
  job_id = util.os_capture("curl -g -s '" .. server .. "lookahead?attr_values=" .. attr_values .. "&attribute=" .. attribute .. "'")
end

function server_reload()
  mode_current_step = { 0 }
  util.os_capture("curl -g -s " .. server .. "reload")
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

    if job_id ~= nil and current_step % 4 == 0 then
      server_sync()
    end

    local notes = get_current_notes()

    local notes_at_step = notes[tostring(current_step-1)]

    if notes_at_step then
      for i, note in pairs(notes_at_step) do
        local pitch = note.pitch

        if drums then
          engine.trig(pitch >= 3 and pitch - 1 or pitch)
        else
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

  if n == 2 then
    mode = (mode % #modes) + 1
    server_lookahead()
  elseif n == 3 then
    server_reload()
    server_lookahead()
  end
end

function enc(n, d)
  -- encoder actions: n = number, d = delta
  if n == 2 then
    mode_current_step[mode] = util.clamp(mode_current_step[mode] + d, mode_min, mode_max)
  elseif n == 3 then
    current_interpolation = util.clamp(current_interpolation + d, 1, interpolation_steps)
  end
  redraw()
end

function draw_sample(sample, offsetX, offsetY)
  screen.pixel(sample['x'] + offsetX, sample['y'] + offsetY)
  screen.fill()
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

  g:all(0)
  local notes = get_current_notes()
  for step = 1, total_steps do
    local notes_at_step = notes[tostring(step-1)]
    if notes_at_step then      
      for i, note in pairs(notes_at_step) do
        pitch = note.pitch
        duration = note.duration
        
        level = (current_step >= step and current_step < step + duration) and 15 or 4
        screen.level(level)
        if drums then
          screen.rect((step - 1) * 2 + 4, 24 + pitch * 3, 2, 2)
          screen.fill()
          if step < 17 and (current_interpolation == 1 or current_interpolation == interpolation_steps) then
            g:led(step, pitch, level)
          end
        else
          screen.move((step - 1) * 2 + 4, 16 - pitch + max_note)
          screen.line_rel(2 * duration, 0)
          screen.stroke()
        end
      end
    end

    g:refresh()
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
    screen.move_rel(6, 0)
  end

  screen.update()
end

local grid = util.file_exists(_path.code.."midigrid") and include "midigrid/lib/mg_128" or grid
g = grid.connect()

g.key = function (x, y, z)
  if (z ~= 1) then return end
  if (current_interpolation ~= 1 and current_interpolation ~= interpolation_steps) then return end
  toggleDrum(x, y)
end

function toggleDrum(x, y)
  local notes = get_current_notes()
  if notes[tostring(x-1)] == nil then
    notes[tostring(x-1)] = {}
  end

  -- if note already exists, toggle off
  for i, note in pairs(notes[tostring(x-1)]) do
    if note.pitch == y then
      notes[tostring(x-1)][i] = nil
      return
    end
  end

  -- toggle on
  local note = { pitch=y, duration=1 }
  table.insert(notes[tostring(x-1)], note)
end