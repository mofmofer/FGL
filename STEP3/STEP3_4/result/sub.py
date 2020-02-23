"""
出力された各機械学習の評価.csvファイルをまとめます.
"""
import subprocess

subprocess.run("ls")
cmd = "cat *.csv > ./join/STEP3_result.csv"
subprocess.check_call(cmd, shell=True)