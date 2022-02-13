-- thesis (placeholder name)
-- jacob zhang

engine.name = 'PolyPerc'
util = require 'util'
json = include('lib/json')
MusicUtil = require "musicutil"
dofile(_path.code .. 'thesis/lib/midi.lua')

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
current_notes = {}

-- attribute vector mode: 1: density, 2: averageInterval
mode = 1
modes = { "DSTY", "AVG_ITVL"}
mode_steps = 7
mode_current_step = { } 
mode_current_notes = { } 


-- interpolation
current_interpolation = 1

samples = {}
density_samples = {}

function server_sample(n)
  return json.parse(util.os_capture('curl -s ' .. server .. 'sample?n=' .. n))
end

function server_sample_and_interpolate()
  return json.parse(util.os_capture('curl -s ' .. server .. 'sample_and_interpolate?n=' .. interpolation_steps))
end

-- function server_interpolate(seq1, seq2)
--   local data = json.stringify({seq1, seq2})
--   local n = interpolation_steps - 2
--   return util.os_capture("curl -s -XPOST -H 'Content-Type: application/json' --data '" .. data .. "' " .. server .. "interpolate?n=" .. n) 
-- end

function server_interpolate(seq1, seq2)
  local hash1 = seq1.hash
  local hash2 = seq2.hash
  local n = interpolation_steps - 2
  return json.parse(util.os_capture("curl -s '" .. server .. "interpolate_existing?hash1=" .. hash1 .. "&hash2=" .. hash2 .. "&n=" .. n .. "'"))
end

function server_attribute_arithmetics(seq, attribute)
  local hash = seq.hash
  local n = interpolation_steps - 2
  return json.parse(util.os_capture("curl -s '" .. server .. "attribute_arithmetics?attribute=density&hash=" .. hash .. "&n=" .. n .. "'"))
end

function process_sample(sample)
  -- print(json.stringify(sample))
  local notes = {}
  seq = sample.notes
  for i = 1, #seq do
    local note = seq[i]
    local start_time = note.start_time
    local duration = note.end_time - note.start_time
    local pitch = note.pitch
    notes[start_time] = { pitch, duration }
  end
  -- print(json.stringify(notes))
  return notes
end

function read_midi(filename)
  local notes = {}
  print(_path.code .. '/thesis/midi/' .. filename .. '.mid')
  local file = io.open(_path.code .. 'thesis/midi/' .. filename .. '.mid', "rb") -- r read mode and b binary mode
  local content = file:read "*a"
  file:close()

  local score = midi2score(content)
  print(json.stringify(score))
  ticks_per_step = score[1] / steps_per_beat
  seq = score[3]
  for i = 1, #seq do
    local note = seq[i]
    if note[1] == 'note' then
      local start_time = note[2]
      local duration = note[3]
      local pitch = note[5]
      notes[math.floor((start_time / ticks_per_step) + 1)] = { pitch, math.floor(duration / ticks_per_step) }
    end
  end
  return notes
end

function generate_and_interpolate()
  server_samples = server_sample_and_interpolate()
  for i = 1, interpolation_steps do
    samples[i] = server_samples[i]
    current_notes[i] = process_sample(server_samples[i])
  end
  density_samples = server_attribute_arithmetics(samples[current_interpolation])
end

function generate()
  current_notes = {}
  local left_key = keys[current_left_key]
  local left_chord = chords[current_left_chord]
  if left_chord == 'M' then left_chord = '' end

  local right_key = keys[current_right_key]
  local right_chord = chords[current_right_chord]
  if right_chord == 'M' then right_chord = '' end

  server_samples = server_sample(2)
  samples[1] = server_samples[1]
  samples[interpolation_steps] = server_samples[2]
  
  current_notes[1] = process_sample(server_samples[1])
  current_notes[interpolation_steps] = process_sample(server_samples[2])

  interpolate()
end

function interpolate()
  local left_key = keys[current_left_key]
  local left_chord = chords[current_left_chord]
  if left_chord == 'M' then left_chord = '' end

  local right_key = keys[current_right_key]
  local right_chord = chords[current_right_chord]
  if right_chord == 'M' then right_chord = '' end

  server_samples = server_interpolate(samples[1], samples[interpolation_steps])
  for i = 1, interpolation_steps - 2 do
    samples[i + 1] = server_samples[i]
    current_notes[i + 1] = process_sample(server_samples[i])
  end
end

function init()
  clock.run(function()
    while initialized == false do
      response_code = util.os_capture('curl -s -o /dev/null -w "%{http_code}" ' .. server)
      print(response_code)
      if response_code == '200' then
        initialized = true
        generate_and_interpolate()
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

    local notes = current_notes[current_interpolation]

    if notes[current_step] then
      local freq, duration = table.unpack(notes[current_step])
      engine.hz(MusicUtil.note_num_to_freq(freq))
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
  elseif n == 3 then
    generate_and_interpolate()
  end
end

function enc(n, d)
  -- encoder actions: n = number, d = delta
  if n == 2 then
    mode_current_step[mode] = util.clamp(mode_current_step[mode] + d, 1, mode_steps)
  elseif n == 3 then
    current_interpolation = util.clamp(current_interpolation + d, 1, interpolation_steps)
  end
  redraw()
end

function redraw()
  screen.clear()

  screen.level(4)
  screen.move(0, 10)
  screen.font_size(8)
  screen.text('THESIS')

  if not initialized then
    screen.move(0,20)
    screen.level(15)
    screen.text('Connecting to server...')
    return
  end

  local notes = current_notes[current_interpolation]
  for step = 1, total_steps do
    if notes[step] then
      pitch, duration = table.unpack(notes[step])

      screen.move((step - 1) * 4, 16 - pitch + max_note)
      screen.line_rel(4 * duration, 0)
      screen.level((current_step >= step and current_step < step + duration) and 15 or 4)
      screen.stroke()
    end
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
    screen.text(mode_name)
    screen.move_rel(6, 0)
  end

  -- attribute vector bar
  if mode_current_step[mode] == nil then mode_current_step[mode] = mode_steps / 2 end

  screen.rect(110, 57, 18 * (mode_current_step[mode] - 1)/(mode_steps - 1), 5)
  screen.level(15)
  screen.fill()

  screen.rect(110, 57, 18, 5)
  screen.level(4)
  screen.stroke()

  screen.update()
end

function cleanup()
  -- deinitialization, kill the server
  -- print('kill ' .. pid)
  -- io.popen('kill ' .. pid)
end
