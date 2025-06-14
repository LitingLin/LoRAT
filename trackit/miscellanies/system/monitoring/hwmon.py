import psutil


class SystemHardwareMonitoring:
    def __init__(self):
        self.sensors = {}

    def update(self):
        if not hasattr(psutil, 'sensors_temperatures'):
            return
        self.sensors = psutil.sensors_temperatures()

    def get_cpu_package_temperature(self):
        if 'coretemp' in self.sensors:
            return self.sensors['coretemp'][0].current
        if 'k10temp' in self.sensors:
            for sensor in self.sensors['k10temp']:
                if sensor.label == 'Tdie':
                    return sensor.current
            return self.sensors['k10temp'][0].current
        if 'zenpower' in self.sensors:
            return self.sensors['zenpower'][0].current
        if 'cpu_thermal' in self.sensors:
            return self.sensors['cpu_thermal'][0].current
        return None
    
    def get_memory_temperature(self):
        if 'spd5118' in self.sensors:
            # take maximum temperature of all memory modules
            return max(sensor.current for sensor in self.sensors['spd5118'])
        return None
