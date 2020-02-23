import subprocess

subprocess.run("ls")
cmd = "cat *.csv > ./join/STEP2_result.csv"
subprocess.check_call(cmd, shell=True)
