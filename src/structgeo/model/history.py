import structgeo.model.geo as geo

""" Collection of useful higher abstraction geo structures."""

class FaultSequence:
    def __init__(self, faults):
        self.faults = faults

    def get_faults(self):
        return self.faults

    def get_fault_count(self):
        return len(self.faults)

    def get_fault(self, index):
        return self.faults[index]

    def get_faults_as_geo_models(self):
        return [geo.Fault(f) for f in self.faults]



