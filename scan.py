
import pyvisa

rm = pyvisa.ResourceManager("@py")
print("Backend:", rm)
resources = rm.list_resources()

print("Resources:", resources[0])
