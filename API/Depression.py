# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:42:25 2023

@author: DELL
"""

from pydantic import BaseModel
# 2. Class which describes the text
class Depresion(BaseModel):
    text: str 
    
    