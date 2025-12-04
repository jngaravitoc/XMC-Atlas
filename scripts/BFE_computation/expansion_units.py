"""
Define units systems here 
"""
import os

def setEXPunits(unit_system): 
    if unit_system == 'gadget':
        units = [
         ('mass', 'Msun', 1e10),
         ('length', 'kpc', 1.0),
         ('velocity', 'km/s', 1.0),
         ('G', 'mixed', 43007.1)
        ]
    else: 
        raise ValueError(f"unit_system {unit_system} not found")
    return units

def setAGAMAunits(unit_system):
    if unit_system == "gadget":
        units = {
            "mass": 1e10,
             "length": 1.0,
             "velocity": 1.0,
              }
    else: 
        raise ValueError(f"unit_system {unit_system} not found")
    return units

