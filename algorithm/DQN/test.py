class Celsius: 
    def __init__(self, temperature=0):
        self.temperature = temperature
    
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
    @property
    def temperature(self):
        return self._temperature
    @temperature.setter
    def temperature(self, value):
        if value < -300:
            print("can not do it")
        else:
            self._temperature = value

    @property
    def unW(self):
        return 1
    

human = Celsius()

# # set the temperature
# human.temperature = -400

# print(human.to_fahrenheit())

print(human)

