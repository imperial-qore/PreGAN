import os
import logging
import json
import re
from subprocess import call

def setup(cfg):
    # For ansible setup
    host = []
    num_hosts = len(cfg["servers"])
    for i in range(num_hosts):
            logging.debug("Creating enviornment with configuration file as  :{}".format(cfg))
            typeID = i%2 # np.random.randint(0,3) # i%3 #
            flavor = instance_type[typeID]
            vm = {}
            vm["flavor"]=flavor
            vm["name"]="SimpeE-Worker-"+str(i)
            host.append(vm)
            cfg["hosts"]=host    
    cfg = json.dumps(cfg)   

    call(["ansible-playbook","playbooks/client.yml","-e",cfg])

# Vagrant setup functions

def setupVagrantEnvironment(cfg, mode):
    with open(cfg, "r") as f:
        config = json.load(f)
    if mode in [0, 1]:
        call(["cd", "framework/config/"])
        with open('Vagrantfile', 'r') as file:
            data = file.read()
        custom_list = "servers=[\n"
        host_ip = []
        for i, datapoint in enumerate(config['vagrant']['servers']):
            custom_list += "\t{\n\t\t:hostname => 'vm"+str(i+1)+"',\n\t\t:ip => '192.168.0."+str(i+2)+"',\n\t\t:box => '"
            custom_list += config['vagrant']['box']
            custom_list += "',\n\t\t:ram => "+str(datapoint['ram'])+",\n\t\t:cpu => "+str(datapoint['cpu'])+"\n\t}"
            host_ip.append("192.168.0."+str(i+2))
            if i != len(config['vagrant']['servers']) - 1:
                custom_list += ","
            custom_list += "\n"
        custom_list += "]\n\n"
        data = re.sub(r"servers=\[((.|\n)*)agent_path=", custom_list+"\nagent_path=", data)
        data = re.sub(r"agent_path=\[((.|\n)*)Vagrant", "agent_path='"+os.getcwd().replace('\\', '/')+"/framework/agent'\n\nVagrant", data)
        with open('Vagrantfile', 'w') as file:
            file.write(data)
        call(["vagrant", "up", "--parallel"])
        call(["cd", "../.."])
        return host_ip
    return ['192.168.0.'+str(i+2) for i in range(len(config['vagrant']['servers']))]    

def destroyVagrantEnvironment(cfg, mode):
    if mode in [0, 3]:
        call(["vagrant", "destroy -f"])
