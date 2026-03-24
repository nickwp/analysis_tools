"""
Module to load detector geometry from a YAML file and compute distances between detectors.
Written by Bruno, to be included by Alie in the general beam analysis code.

Added a second class to read the run information stored in the json files at /eos/experiment/wcte/configuration/run_info/google_sheet_beam_data.json
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union
import yaml
import json


class ReadBeamRunInfo:
    """Reads in the run information stored in the json file"""
    def __init__(self, no_acts = False):
        #We have changed the place where the google sheet json is stored because of permission issues, will revert back soon, 
        with open("/eos/experiment/wcte/configuration/run_info/google_sheet_beam_data.json") as f:
            self.runs = json.load(f)
        self.is_no_acts = no_acts

    def get_info_run_number(self, run_number):

        target_run = None
        for run in self.runs:
            if run.get("run_number") == str(run_number):
                self.target_run = run
                target_run = run
        if target_run == None:
            raise Exception(f"The run {run_number} was not found in the /eos/experiment/wcte/configuration/run_info/google_sheet_beam_data.json referecence file")

        run_number = int(target_run.get("run_number"))
        run_momentum = int(target_run.get("beam_momentum"))

        if target_run.get("act0")=="out":
            raise Exception(f"There is no ACT0 in run {run_number}")

        

        

        there_is_ACT5 = True
        
        #in case there are no ACT3, ACT0 values we set to the smallest refractive index 
        try:
            n_eveto_group = float(target_run.get("act0"))
        except:
            print(f"The ACT0 refractive index is {target_run.get("act0")}, instead we set n_eveto as 1.01")
            n_eveto_group = 1.01
            

        try:
            n_tagger_group = float(target_run.get("act3"))
        except:
            print(f"The ACT3 refractive index is {target_run.get("act3")}, instead we set n_tagger as 1.01")
            n_tagger_group = 1.01
            
            
        
            
         
        if self.is_no_acts == True:
            print("Careful, we are in a minimum biais run, without downstream acts, we are over-ridding the usual requirements for ACT matching")


        if (target_run.get("act5")=="out") or (target_run.get("act5")=="OUT"):
            there_is_ACT5 = False
            
            
        if (target_run.get("lead_glass")=="IN"):
            raise Exception(f"This beam analysis code is designed for runs where the lead_glass is out of the beamline, in run {run_number} it is {target_run.get('lead_glass')}")
            
            
        if self.is_no_acts == False:
            
            if (target_run.get("act0") != target_run.get("act1") or target_run.get("act0")!= target_run.get("act2") or target_run.get("act1") != target_run.get("act2")):
                raise Exception("The three upstream ACTs should have the same refractive index")
            
            if (target_run.get("act3")=="out") or (target_run.get("act3")=="OUT"):
                raise Exception(f"This beam analysis code is designed for runs where there are at least ACT3 and ACT4 in the beamline, in this run there are no ACT3, we are not runing the beam code for it")

            if (target_run.get("act4")=="out") or (target_run.get("act4")=="OUT"):
                raise Exception(f"This beam analysis code is designed for runs where there are at least ACT3 and ACT4 in the beamline, in this run there are no ACT4, we are not runing the beam code for it")

            if there_is_ACT5:
                if (target_run.get("act3") != target_run.get("act4") or target_run.get("act3")!= target_run.get("act5") or target_run.get("act4") != target_run.get("act5")):
                    raise Exception("The three downstream ACTs should have the same refractive index")
            else:
                if (target_run.get("act3") != target_run.get("act4")):
                    raise Exception("The two downstream ACTs should have the same refractive index")

        beam_config = target_run.get("beam_config")

        return run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, beam_config


    def print_run_summary(self, there_is_ACT5):
        run = self.target_run
        print("\n" + "="*60)
        print(f" Run summary")
        print("="*60)

        print(f" Run number        : {run.get('run_number')}")
        print(f" Beam momentum     : {run.get('beam_momentum')} MeV/c")
        print(f" Beam config       : {run.get('beam_config')}")
        print(f" Trigger config    : {run.get('trigger_config')}")
        print(f" Lead glass        : {run.get('lead_glass')}")

        print("\n ACT configuration")
        print(" -----------------")
        print(f" Upstream ACTs     : n = {run.get('act0')} (ACT0/1/2)")
        print(f" Downstream ACTs   : n = {run.get('act3')} (ACT3/4"
              f"{'/5' if there_is_ACT5 else ''})")

        print("\n Timing")
        print(" ------")
        print(f" Date              : {run.get('date')}")
        print(f" Start time        : {run.get('start_time')}")
        print(f" End time          : {run.get('end_time')}")

        print("\n Status / comments")
        print(" -----------------")
        print(f" Status            : {run.get('status')}")
        comments = run.get("comments")
        if comments:
            print(f" Comments          : {comments}")

        print("="*60 + "\n")



@dataclass(frozen=True)
class DetectorDB:
    """
    Container for detector geometry information loaded from a YAML file.
    frozen=True means the content cannot be modified after creation,
    """
    data: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DetectorDB":
        """
        Reads a YAML file and returns a DetectorDB instance.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file containing detector definitions.

        Returns
        -------
        DetectorDB
            A DetectorDB object holding the parsed YAML data.

        Raises
        ------
        ValueError
            If the YAML file does not contain a valid 'detectors' section.
        """
        path = Path(path)

        # Open and parse the YAML file
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Basic validation to ensure expected structure exists
        if not isinstance(data, dict) or "detectors" not in data:
            raise ValueError("YAML must contain a top level key named 'detectors'.")

        if not isinstance(data["detectors"], dict) or not data["detectors"]:
            raise ValueError("'detectors' must be a non empty mapping of detector names.")

        # Create and return the DetectorDB object
        return cls(data=data)

    def _center(self, det_name: str) -> float:
        """
        Internal helper method to retrieve the center position of a detector.

        Parameters
        ----------
        det_name : str
            Name of the detector (e.g. 'T0', 'T4', 'T5').

        Returns
        -------
        float
            Center position of the detector in meters.

        Raises
        ------
        KeyError
            If the detector name is not defined in the YAML.
        ValueError
            If the detector does not define 'center_m'.
        """
        dets = self.data["detectors"]

        # Check that the requested detector exists
        if det_name not in dets:
            available = ", ".join(sorted(dets.keys()))
            raise KeyError(f"Unknown detector '{det_name}'. Available: {available}")

        # Extract the center position
        center = dets[det_name].get("center_m", None)
        if center is None:
            raise ValueError(f"Detector '{det_name}' is missing 'center_m'.")

        return float(center)

    def _thickness(self, det_name: str) -> float:
        """
        Internal helper method to retrieve the thickness of a TS detector

        Parameters
        ----------
        det_name : str
            Name of the detector (e.g. 'T0', 'T4', 'T5').

        Returns
        -------
        float
            Thickness of the detector in meters.

        Raises
        ------
        KeyError
            If the detector name is not defined in the YAML.
        ValueError
            If the detector does not define 'center_m'.
        """
        dets = self.data["detectors"]

        # Check that the requested detector exists
        if det_name not in dets:
            available = ", ".join(sorted(dets.keys()))
            raise KeyError(f"Unknown detector '{det_name}'. Available: {available}")

        # Extract the center position
        layers = dets[det_name].get("layers_m", None)
        if layers is None:
            raise ValueError(f"Detector '{det_name}' is missing 'layers_m'.")

        return layers

    def distance_m(self, a: str, b: str) -> float:
        """
        Compute the absolute distance between two detectors.

        The order of detectors does not matter:
        distance_m('T0', 'T5') == distance_m('T5', 'T0')

        Parameters
        ----------
        a, b : str
            Names of the two detectors.

        Returns
        -------
        float
            Absolute distance between detector centers in meters.
        """
        return abs(self._center(a) - self._center(b))


    def get_thickness_m(self, det:str, mat:str) -> float:
        """
        Returns the thickness of a given material (mat) layer for a detector (det) in units of meter

        Parameters
        ----------
        det: the detector name
        mat: the material name, should be "scintillator"; "mylar" or "vinyl" for TS detectors

        """
        thickness_dict = self._thickness(det)

        if mat not in thickness_dict.keys():
            raise ValueError(f"Material '{mat}' is missing in 'layers_m': {thickness_dict} for detector {det}.")

        return thickness_dict[mat]

    def get_total_thickness_m(self, det:str) -> float:
        """
        Returns the total thickness of detector (det) in units of meter

        Parameters
        ----------
        det: the detector name

        """
        thickness_dict = self._thickness(det)

        tot_thickness = 0

        #this is only including one layer of mylar and of vinyl, incorrect
        for mat in thickness_dict.keys():
            tot_thickness += thickness_dict[mat]



        return tot_thickness


    def get_all_layers(self, det:str):
        """Returns three arrays, the name of each layers, the thickness of each layer and the material of each layer."""

        upstream  = True
        if det[0:3] == "ACT":
            #we are removing the number of the ACT detector to reduce the number of entries in the yaml file
            layers = self._thickness(det[0:3]+det[4:])
            #check if this is an upstream or a downstream ACT
            if int(det[3]) > 2:
                upstream = False 
            

        else:
            layers = self._thickness(det)

        
        layer_names = []
        layer_materials = []
        layer_thicknesses = []
        
        
        for material in layers.keys():
            
            layer_names.append(det+"_"+material)
            layer_thicknesses.append( layers[material])
            
            if material == "aerogel":
                if upstream:
                    layer_materials.append("upstreamAerogel")
                else:
                    layer_materials.append("downstreamAerogel")
                    
                    
            elif material == "scintillator":
                layer_materials.append("scintillator")
                
            elif material[0:3] == "air":
                layer_materials.append("air")
                
            elif material[0:5] == "vinyl":
                layer_materials.append("vinyl")
                
            elif material[0:5] == "mylar":
                layer_materials.append("mylar")
                
            else:
                raise Exception(f"The material {material} is not recognised")


        return layer_names, layer_thicknesses, layer_materials

def detector_distance_m(yaml_path: Union[str, Path], det_a: str, det_b: str) -> float:
    """
    Convenience function to compute detector distance in one line.

    This is useful for scripts or notebooks where you do not want
    to explicitly manage a DetectorDB object.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the detector YAML file.
    det_a, det_b : str
        Names of the two detectors.

    Returns
    -------
    float
        Absolute distance between detector centers in meters.
    """
    db = DetectorDB.from_yaml(yaml_path)
    return db.distance_m(det_a, det_b)


if __name__ == "__main__":
    # Example usage when running this file directly

    db = DetectorDB.from_yaml("../include/wcte_beam_detectors.yaml")

    print("T0 to T1 distance (m):", db.distance_m("T0", "T1"))
    print("T0 to T4 distance (m):", db.distance_m("T0", "T4"))
    print("T0 to T5 distance (m):", db.distance_m("T0", "T5"))
    print("T4 to T1 distance (m):", db.distance_m("T4", "T1"))
    print("T4 to T5 distance (m):", db.distance_m("T4", "T5"))
    print("T1 to T5 distance (m):", db.distance_m("T1", "T5"))
