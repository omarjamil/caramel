import configparser
import subprocess

# This parses the rose_app.conf file, finds umstash_streq sections and then
# filters out dom_name DIAG profiles as I am interested in the surface vars
if __name__ == "__main__":
    stash_codes = []
    config = configparser.ConfigParser()
    config.read('nested_rose-app.conf')
    
    for s in config.sections():
        if 'umstash_streq' in s:
            if 'DIAG' in config[s]['dom_name']:
                stash_code = '{0}{1}'.format((config[s]['isec']).zfill(2),(config[s]['item']).zfill(3))
                stash_codes.append(stash_code)

    for s in stash_codes:
        command = '/opt/ukmo/utils/bin/stash -c '+s
        subprocess.call(command, shell=True)

                
