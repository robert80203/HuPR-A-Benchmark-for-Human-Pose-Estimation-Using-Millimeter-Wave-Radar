import yaml

# with open("environment.yml") as file_handle:
#     environment_data = yaml.load(file_handle, Loader=yaml.Loader)
#     print(len(environment_data))
with open("environment.yml", "r") as stream:
    environment_data = yaml.load(stream, Loader=yaml.Loader)
    print(type(environment_data["dependencies"]))
    
with open("requirements.txt", "w") as file_handle:
    
    for dependency in environment_data["dependencies"]:
        # print(dependency)
        # print(dependency_tuples)
        package_name, package_version,__ = dependency.split("=")
        file_handle.write("{}=={}\n".format(package_name, package_version))
        
