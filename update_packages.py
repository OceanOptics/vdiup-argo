import subprocess

# List all installed packages
subprocess.run(['pip', 'list'])

# Check for outdated packages
outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf-8')
print(outdated_packages)

# Extract package names from the outdated packages list
package_names = [line.split()[0] for line in outdated_packages.split('\n')[2:] if line]

# Update each outdated package
for package in package_names:
    subprocess.run(['pip', 'install', '--upgrade', package])