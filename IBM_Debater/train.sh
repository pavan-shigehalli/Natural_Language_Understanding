#!/bin/bash
# runs the ML program.
# monitors the CPU temperature and system memory
# in case of critical temperature, program is paused and
# resumed  after the temperature returns to a normal value
# in case of complete memory usage program is shut down and
# restarted after some time

script_path=$(dirname "$0")
log_file='logs/error.log'
py_prg='train_lstm_tri_net.py'

max_temp=70 # temeprature in celcius
avg_temp=35 # temeprature in celcius
min_memory=1000000 # minimum free memory in MB
critical_temperature=false
program_aborted=false
# start the script
echo "[$(date) ] Program is starting" >> $script_path/$log_file
python3 $script_path/$py_prg >>  $script_path/$log_file &
sleep 30

while true;do
  # CPU temperature monitoring
  # get core temperature
  temp0=$(($(cat /sys/class/thermal/thermal_zone0/temp) / 1000))
  temp1=$(($(cat /sys/class/thermal/thermal_zone1/temp) / 1000))
  temp2=$(($(cat /sys/class/thermal/thermal_zone2/temp) / 1000))

  if [ ! $critical_temperature = true ]; then
    if [ $temp0 -gt $max_temp ] || [ $temp1 -gt $max_temp ] || [ $temp2 -gt $max_temp ]; then
      pid=$(pgrep -f $py_prg)
      if [ $pid ]; then
        # pause the program
        kill -STOP $pid
        echo "[$(date) ] CRITICAL CPU TEMPERATURE. PROGRAM WILL BE RESUMED AFTER CPU COOLS DOWN" >> $script_path/$log_file
      fi
      critical_temperature=true
      program_aborted=true
    fi
  fi

  if [ $program_aborted = true ]; then
    if [ $critical_temperature = true ]; then
      if [ $temp0 -lt $avg_temp ] || [ $temp1 -lt $avg_temp ] || [ $temp2 -lt $avg_temp ]; then
        critical_temperature=false
        program_aborted=false
        echo "[$(date) ] Program is resuming" >> $script_path/$log_file
        # Resume the program
        kill -CONT $pid
      fi
    fi
  else
    pid=$(pgrep -f $py_prg)
    if [ ! $pid ]; then # program is aborted by another application
      echo "[$(date) ] program is killed by an external process" >> $script_path/$log_file
      sleep 2
      # Restart the program
      echo "[$(date) ] Program is restarting" >> $script_path/$log_file
      python3 $script_path/$py_prg >>  $script_path/$log_file &
    fi
  fi

  # Memory monitoring
  free_memory=$(awk '/MemAvailable/ { print $2 }' /proc/meminfo)
  if [ $free_memory -lt $min_memory ]; then
    pid=$(pgrep -f $py_prg)
    if [ $pid ]; then
      kill $pid
      echo "[$(date) ] NOT ENOUGH MEMORY. PROGRAM WILL BE RESTARTED" >> $script_path/$log_file
      sleep 15
    fi
    # Restart the program
    echo "[$(date) ] Program is restarting" >> $script_path/$log_file
    python3 $script_path/$py_prg >>  $script_path/$log_file &
  fi

  sleep 30
done
