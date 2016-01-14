import exceptions
import re


import numpy as np

def duck (instr):
    if "," in instr:
        try:
            return [float(s) for s in instr.split(',')]
        except:
            return instr.strip("'")
    try:
        return int(instr)
    except exceptions.ValueError:
        try:
            return float(instr)
        except exceptions.ValueError:
            return instr.strip("'")

def parse(filename):
    class struct:
        pass
    struct.top = {}
    curstr = 'top'
    for l in open(filename):
        cmd = l.split('!')[0].strip()
        if cmd:
            if "&" in cmd:
                dlm =  cmd.strip().strip('&')
                struct.__dict__[dlm] = {}
                curstr = dlm
            elif "=" in cmd:
                var,arg = cmd.split("=")
                arg = arg.strip().rstrip(',').rstrip('/')
                var = var.strip()
                if "(" and ")" in var:
                    ind = int(var[var.index("(") + 1:var.rindex(")")])
                    listvar = var.split('(')[0].strip()
                    if not listvar in struct.__dict__:
                        setattr(struct,listvar,defaultlist())
                    struct.__dict__[listvar][ind] = duck(arg)
                    struct.__dict__[curstr][listvar] = struct.__dict__[listvar]
                else:
                    struct.__dict__[curstr][var] = duck(arg)
                    setattr(struct, var,duck(arg))

    return struct


class defaultlist(list):

   def __setitem__(self, index, value):
      size = len(self)
      if index >= size:
         self.extend(0 for _ in range(size, index + 1))

      list.__setitem__(self, index, value)
