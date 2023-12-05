import subprocess
import os

for i in range(100):

    subprocess.run(['/usr/bin/python3', 'train.py'])
    subprocess.run(['/usr/bin/python3', 'test.py'])

    os.system("clear")
    print(f"Progress: {(i + 1)}%")

print("Finished")