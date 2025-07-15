import os
import sys
import json
import yaml
import warnings
import logging

import numpy as np
import awkward as ak
import vector

from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import candidate
from coffea.util import load, save
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask

import hist
from optparse import OptionParser
from omegaconf import OmegaConf

from base_class.hist import Fill

from analysis.helpers.common import apply_jerc_corrections, update_events

from bbww.analysis.helpers.ids import (
    lepton_selection,
    tau_selection,
    photon_selection,
    jet_selection,
    HEMjet_selection,
)
from bbww.analysis.helpers import common
import bbww.analysis.helpers.corrections as corrections

warnings.filterwarnings("ignore", "Missing cross-reference index for")
warnings.filterwarnings("ignore", "Please ensure")
warnings.filterwarnings("ignore", "invalid value")

vector.register_awkward()

class analysis(processor.ProcessorABC):
    def __init__(
        self,
        path: str = "bbww/analysis/data",
        parameters: str = "bbww/analysis/metadata/object_preselection.yaml",
        corrections_metadata: str = "analysis/metadata/corrections.yml",
    ):
        self.path = path
        self.parameters = parameters
        with open(corrections_metadata, "r") as f:
            self.corrections_metadata = yaml.safe_load(f)


    def update(events, collections):
        """Return a shallow copy of events array with some collections swapped out"""
        out = events
        for name, value in collections.items():
            out = ak.with_field(out, value, name)
        return out

    # def make_output():
    #     return {
    #         'nEvents': {},
    #         'cutflow': {},
    #         # 'test': {},
    #         # 'hists' : {
    #         #     "met": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 300, name="met", label = 'pT [GeV]')  
    #         #         .StrCat([], name="systematic", growth = True)      
    #         #         .StrCat([], name="dataset", growth = True)     
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False) 
    #         #         .Weight()                               
    #         #     ),
    #         #     "chi_hadW": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 5, name="chi_hadW", label=r'$\chi^2$')
    #         #         .StrCat([], name="dataset", growth = True)
    #         #         .StrCat([], name="systematic", growth = True)  
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False)
    #         #         .Weight()
    #         #     ),
    #         #     "chi_hadWs": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 5, name="chi_hadWs", label=r'$\chi^2$')
    #         #         .StrCat([], name="systematic", growth = True)  
    #         #         .StrCat([], name="dataset", growth = True)
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False)
    #         #         .Weight()
    #         #     ),
    #         #     "chi_tt": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 5, name="chi_tt", label=r'$\chi^2$')
    #         #         .StrCat([], name="systematic", growth = True)  
    #         #         .StrCat([], name="dataset", growth = True)
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False)
    #         #         .Weight()
    #         #     ),
    #         # }
    #     }

    def process(self, events):

        logging.debug(f"Metadata: {events.metadata}\n")
        self.dataset = events.metadata['dataset']
        self.year = events.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = events.metadata['processName']
        self.isData = not hasattr(events, "genWeight")
        self.isMC = not self.isData
        

        # jets = apply_jerc_corrections(
        #     events.Jet,
        #     corrections_metadata=self.corrections_metadata[self.year],
        #     isMC=self.isMC,
        #     run_systematics=False, ###self.run_systematics,
        #     dataset=self.dataset
        # )

        shifts = [({"Jet": events.Jet}, None)]
        #need to work on a new way of loading corrections centrally from CMS
        '''corrections = load(f'{path}/corrections.coffea')

        jet_factory              = corrections['jet_factory']
        met_factory              = corrections['met_factory']

        nojer = "NOJER" if skipJER else ""
        if 'year' in events.metadata:
            year = events.metadata['year'].replace('UL','20').replace("_", "")
        thekey = f"{year}mc{nojer}"

        def add_jec_variables(jets, event_rho):
            jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
            jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
            return jets
        
        jets = jet_factory[thekey].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll))
        met = met_factory.build(events.MET, jets)

        shifts = [({"Jet": jets,"MET": met}, None)]
        if systematics:
            shifts.extend([
                ({"Jet": jets.JES_jes.up, "MET": met.JES_jes.up}, "JESUp"),
                ({"Jet": jets.JES_jes.down, "MET": met.JES_jes.down}, "JESDown"),
                ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.up}, "UESUp"),
                ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.down}, "UESDown"),
            ])
            if not skipJER:
                shifts.extend([
                    ({"Jet": jets.JER.up, "MET": met.JER.up}, "JERUp"),
                    ({"Jet": jets.JER.down, "MET": met.JER.down}, "JERDown"),
                ])

        # fill histograms separately for each systematic key
        for collections, name in shifts:
            selection(update(events, collections), output, name)'''

        weights = Weights(None, storeIndividual=True)
        list_weight_names = []

        return processor.accumulate( self.process_shift(update_events(events, collections), name, weights, list_weight_names) for collections, name in shifts )

    def process_shift(self, events, shift_name, weights, list_weight_names):

        output = {}
        selection = PackedSelection(dtype="uint64")
        year = events.metadata['year']
        scale = 1 if self.isData else 1000.*float(events.metadata['lumi'])*events.metadata['xs']

        events.metadata['genEventSumw'] = events.metadata.get('genEventSumw', 1.0)
        if self.isMC: weights.add('xsec', scale*events.genWeight/events.metadata['genEventSumw'])

        params = OmegaConf.load(self.parameters)
                
        deepflavWPs = common.common['btagWPs']['deepflav'][self.year_label]
        deepcsvWPs =  common.common['btagWPs']['deepcsv'][self.year_label]

        ###
        #Initialize global quantities (MET ecc.)
        ###

        npv = events.PV.npvsGood 
        run = events.run
        met = events.MET
        met['pt'] , met['phi'] = corrections.get_met_xy_correction(f"{self.year_label}_UL", npv, run, met.pt, met.phi, self.isData)

        ###
        #Initialize physics objects
        ###


        events.Muon['isloose'] = lepton_selection(events, "Muon", params, "loose")
        events.Muon['id_sf'] = ak.where(
            events.Muon.isloose, 
            corrections.get_mu_loose_id_sf(f"{self.year_label}_UL", abs(events.Muon.eta), events.Muon.pt), 
            ak.ones_like(events.Muon.pt)
        )
        events.Muon['iso_sf'] = ak.where(
            events.Muon.isloose, 
            corrections.get_mu_loose_iso_sf(f"{self.year_label}_UL", abs(events.Muon.eta), events.Muon.pt), 
            ak.ones_like(events.Muon.pt)
        )
        events.Muon['istight'] = lepton_selection(events, "Muon", params, "tight")
        events.Muon['id_sf'] = ak.where(
            events.Muon.istight, 
            corrections.get_mu_tight_id_sf(f"{self.year_label}_UL", abs(events.Muon.eta), events.Muon.pt), 
            events.Muon.id_sf
        )
        events.Muon['iso_sf'] = ak.where(
            events.Muon.istight, 
            corrections.get_mu_tight_iso_sf(f"{self.year_label}_UL", abs(events.Muon.eta), events.Muon.pt), 
            events.Muon.iso_sf
        )
        mu_loose = events.Muon[events.Muon.isloose]
        mu_tight = events.Muon[events.Muon.istight]
        mu_ntot = ak.num(events.Muon, axis=1)
        mu_nloose = ak.num(mu_loose, axis=1)
        mu_ntight = ak.num(mu_tight, axis=1)
        leading_mu = ak.firsts(mu_tight)


        events.Electron['isclean'] = ak.all(events.Electron.metric_table(mu_loose) > 0.3, axis=2)
        events.Electron['reco_sf'] = ak.where(
            (events.Electron.pt<20),
            corrections.get_ele_reco_sf_below20(f"{self.year_label}_UL", events.Electron.eta+events.Electron.deltaEtaSC, events.Electron.pt), 
            corrections.get_ele_reco_sf_above20(f"{self.year_label}_UL", events.Electron.eta+events.Electron.deltaEtaSC, events.Electron.pt)
        )
        events.Electron['isloose'] = lepton_selection(events, "Electron", params, "loose")
        events.Electron['id_sf'] = ak.where(
            events.Electron.isloose,
            corrections.get_ele_loose_id_sf(f"{self.year_label}_UL", events.Electron.eta+events.Electron.deltaEtaSC, events.Electron.pt),
            ak.ones_like(events.Electron.pt)
        )

        events.Electron['istight'] = lepton_selection(events, "Electron", params, "tight")
        events.Electron['id_sf'] = ak.where(
            events.Electron.istight,
            corrections.get_ele_tight_id_sf(f"{self.year_label}_UL", events.Electron.eta+events.Electron.deltaEtaSC, events.Electron.pt),
            events.Electron.id_sf
        )
        e_clean = events.Electron[events.Electron.isclean]
        e_loose = e_clean[e_clean.isloose]
        e_tight = e_clean[e_clean.istight]
        e_ntot = ak.num(events.Electron, axis=1)
        e_nloose = ak.num(e_loose, axis=1)
        e_ntight = ak.num(e_tight, axis=1)
        leading_e = ak.firsts(e_tight)

        ## AGE: do we really need taus?

        events.Tau['isclean'] = (
            ak.all(events.Tau.metric_table(mu_loose) > 0.4, axis=2)
            & ak.all(events.Tau.metric_table(e_loose) > 0.4, axis=2)
        )
        events.Tau['isloose'] = tau_selection(events, params, "loose")
        tau_clean = events.Tau[events.Tau.isclean]
        tau_loose = tau_clean[tau_clean.isloose]
        tau_ntot = ak.num(events.Tau, axis=1)
        tau_nloose = ak.num(tau_loose, axis=1)

        ## AGE: do we really need photons?

        events.Photon['isclean'] = (
            ak.all(events.Photon.metric_table(mu_loose) > 0.5, axis=2)
            & ak.all(events.Photon.metric_table(e_loose) > 0.5, axis=2)
            & ak.all(events.Photon.metric_table(tau_loose) > 0.5, axis=2)
        )
        events.Photon['isloose'] = photon_selection(events, params, "loose")
        pho_clean = events.Photon[events.Photon.isclean]
        pho_loose = pho_clean[pho_clean.isloose]
        pho_ntot = ak.num(events.Photon, axis=1)
        pho_nloose = ak.num(pho_loose, axis=1)


        events.Jet['isclean'] = (
            ak.all(events.Jet.metric_table(mu_loose) > 0.4, axis=2)
            & ak.all(events.Jet.metric_table(e_loose) > 0.4, axis=2)
            & ak.all(events.Jet.metric_table(tau_loose) > 0.4, axis=2)
            & ak.all(events.Jet.metric_table(pho_loose) > 0.4, axis=2)
        )
        events.Jet['issoft'] = jet_selection(events, params, year)
        events.Jet['isHEM'] = HEMjet_selection(events)
        j_clean = events.Jet[events.Jet.isclean]
        j_soft = j_clean[j_clean.issoft]
        j_HEM = events.Jet[events.Jet.isHEM]
        j_nsoft = ak.num(j_soft, axis=1)
        j_nHEM = ak.num(j_HEM, axis=1)
        leading_j = ak.firsts(j_clean)

        j_candidates = j_soft[ak.argsort(j_soft.particleNetAK4_QvsG, axis=1, ascending=False)]
        j_candidates = j_candidates[:, :5]
        j_candidates = j_candidates[ak.argsort(j_candidates.particleNetAK4_B, axis=1, ascending=False)]

        valid_jets = ak.num(j_candidates) >= 4
        j_candidates = ak.mask(j_candidates, valid_jets)

        jb_candidates = ak.pad_none(j_candidates[:, :2], 2, axis=1)
        j_candidates = j_candidates[:, 2:]
        j_candidates = j_candidates[ak.argsort(j_candidates.pt, axis=1, ascending=False)]

        jj_i = ak.argcombinations(j_candidates, 2, fields=["j1", "j2"])
        jj_i = jj_i[abs(j_candidates[jj_i.j1].eta - j_candidates[jj_i.j2].eta) < 2.0]
        jj_i = jj_i[(j_candidates[jj_i.j1] + j_candidates[jj_i.j2]).mass < 120.0]
        jj_tt_mask = ak.pad_none(j_candidates[jj_i.j2].pt > 20.0, 3, axis=1)

        qq = ak.pad_none(j_candidates[jj_i.j1] + j_candidates[jj_i.j2], 3, axis=1)
        mbb = (jb_candidates[:, 0] + jb_candidates[:, 1]).mass
            
        #ttbar and hadronic W neutrino pz reconstruction
        def nu_pz(l,v):
            m_w = 80.379
            m_l = l.mass            
            A = (l.px*v.pt * np.cos(v.phi)+l.py*v.pt * np.sin(v.phi)) + (m_w**2 - m_l**2)/2
            B = l.energy**2*((v.pt * np.cos(v.phi))**2+(v.pt * np.sin(v.phi))**2)
            C = l.energy**2 - l.pz**2
            discriminant = (2 * A * l.pz)**2 - 4 * (B - A**2) * C
            # avoiding imaginary solutions
            sqrt_discriminant = ak.where(discriminant >= 0, np.sqrt(discriminant), np.nan)
            pz_1 = (2*A*l.pz + sqrt_discriminant)/(2*C)
            pz_2 = (2*A*l.pz - sqrt_discriminant)/(2*C)
            return ak.where(abs(pz_1) < abs(pz_2), pz_1, pz_2)
            
        # hadronic W* signal reconstruction
        
        # need "charge" here so we can add them with electrons/muons four vectors
        v_e = ak.zip(
            {
                "x": met.pt * np.cos(met.phi),
                "y": met.pt * np.sin(met.phi),
                "z": nu_pz(leading_e, met),
                "t": np.sqrt(met.pt**2 + nu_pz(leading_e, met)**2),
                "charge" : met.pt * 0 
            },
            with_name="Candidate"
        )

        v_mu = ak.zip(
            {
                "x": met.pt * np.cos(met.phi),
                "y": met.pt * np.sin(met.phi),
                "z": nu_pz(leading_mu, met),
                "t": np.sqrt(met.pt**2+nu_pz(leading_mu, met)**2) ,
                "charge" : met.pt * 0 
            },
            with_name="Candidate"
        )

        v_mu = ak.mask(v_mu, ~np.isnan(v_mu.pz))
        v_e = ak.mask(v_e, ~np.isnan(v_e.pz)) #avoid calculations for imaginary solutions

        # H -> lvqq with electrons and muons
        mevqq = (leading_e + v_e + qq ).mass
        mmuvqq = (leading_mu + v_mu + qq).mass

        l_mu = ~ak.is_none(leading_mu.pt)
        l_e = ~ak.is_none(leading_e.pt)
        muge = leading_mu.pt > leading_e.pt

        mlvqq_hadWs = {
            'esr'  : mevqq,
            'msr'  : mmuvqq
        }
            
        mlvqq_hadWs = ak.where(l_mu & l_e,
                            ak.where(muge, mlvqq_hadWs['msr'], mlvqq_hadWs['esr']),
                            ak.where(l_mu, mlvqq_hadWs['msr'], mlvqq_hadWs['esr'])
                            ) #select leading lepton combination

        def chi_square(data,mean,std):
            x_2 = ak.sum(data**2)
            n = ak.count(data[~ak.is_none(data)])
            chi2 = ((data - mean)/std)**2
            return chi2, mean, std

        #individual chi squares for hadronic W* signal selection
        chi1_hadWs, mean1_hadWs, std1_hadWs = chi_square(mbb,116.02, 45.04) # H -> bb            
        chi2_hadWs, mean2_hadWs, std2_hadWs = chi_square(mlvqq_hadWs, 173.59, 48.67) # H -> lvqq
        chi3_hadWs, mean3_hadWs, std3_hadWs = chi_square(qq.mass,41.77, 14.92) #hadronic W*    

        #total chi square
        chi_sq_hadWs = np.sqrt(chi1_hadWs + chi2_hadWs + chi3_hadWs)
        min_chi_sq_hadWs = ak.argmin(chi_sq_hadWs, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
        chi_sq_hadWs = chi_sq_hadWs[min_chi_sq_hadWs]

        jj_gen_mass = ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass, 3, axis=1) #gen mass of dijet pair
        jj_sel_gen_mass_hadWs =  ak.fill_none(ak.firsts(jj_gen_mass[min_chi_sq_hadWs]),-1) #get gen mass of pair selected using chi square

        ## end hadronic W* signal reconstruction

        ## hadronic W signal reconstruction
        
        #transverse mass
        def deltaphi(phi1, phi2):
            dphi = phi1 - phi2
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            return dphi

        mT = {
            'esr'  : np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(deltaphi(met.phi, leading_e.phi)))),
            'msr'  : np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(deltaphi(met.phi, leading_mu.phi))))
        }


        mT_leading_lep = ak.where(l_mu & l_e,
                            ak.where(muge, mT['msr'], mT['esr']),
                            ak.where(l_mu, mT['msr'], mT['esr'])
                            )


        #individual chi squares for hadronic W signal selection
        chi1_hadW, mean1_hadW, std1_hadW = chi_square(mbb,115.33, 46.29) # H -> bb
        chi2_hadW, mean2_hadW, std2_hadW = chi_square(mT_leading_lep, 58.87, 37.35) #transverse mass             
        chi3_hadW, mean3_hadW, std3_hadW = chi_square(qq.mass,66.89, 10.98) #hadronic W

        chi_sq_hadW = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW)
        min_chi_sq_hadW= ak.argmin(chi_sq_hadW, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
        chi_sq_hadW = chi_sq_hadW[min_chi_sq_hadW]

        jj_sel_gen_mass_hadW =  ak.fill_none(ak.firsts(jj_gen_mass[min_chi_sq_hadW]),-1) #gen mass of the di jet pair with minimum chi square
        
        ## ttbar reconstruction
        
        #leptonic top with electrons
        mevb1 = (leading_e + v_e + ak.pad_none(jb_candidates,2,axis=1)[:,0]).mass
        mevb2 = (leading_e + v_e + ak.pad_none(jb_candidates,2,axis=1)[:,1]).mass

        #leptonic top with muons
        mmvb1 = (leading_mu + v_mu + ak.pad_none(jb_candidates,2,axis=1)[:,0]).mass
        mmvb2 = (leading_mu + v_mu + ak.pad_none(jb_candidates,2,axis=1)[:,1]).mass

        mlvb1 = ak.where(l_mu & l_e,
                            ak.where(muge, mmvb1, mevb1),
                            ak.where(l_mu, mmvb1, mevb1)
                            ) #leptonic candidate 1

        mlvb2 = ak.where(l_mu & l_e,
                            ak.where(muge, mmvb2, mevb2),
                            ak.where(l_mu, mmvb2, mevb2)
                            ) #leptonic candidate 2  

        mbqq1 = ak.pad_none((ak.pad_none(jb_candidates,2,axis=1)[:,0] + ak.mask(qq, jj_tt_mask)).mass,3,axis=1) #hadronic candidate 1
        mbqq2 = ak.pad_none((ak.pad_none(jb_candidates,2,axis=1)[:,1] + ak.mask(qq, jj_tt_mask)).mass,3,axis=1) #hadronic candidate 2

        def distance(x1,y1,x2,y2):
            return np.sqrt((x2-x1)**2+(y2-y1)**2)
        
        #ttbar candidates
        tt1 = ak.cartesian({"t1":mlvb1,"t2":mbqq2},axis=1)
        tt2 = ak.cartesian({"t1":mlvb2,"t2":mbqq1},axis=1)
        b_sel = abs(distance(tt1.t1,tt1.t2,172.5,172.5)) <  abs(distance(tt2.t1,tt2.t2,172.5,172.5)) #pick pair closest to ttbar mass

        #conditions to work around None values
        c1 = ~ak.is_none(distance(tt1.t1, tt1.t2,172.5,172.5))
        c2 = ~ak.is_none(distance(tt2.t1, tt2.t2,172.5,172.5))

        #final ttbar candidates
        tt = ak.pad_none(ak.where( c1 & c2, ak.where(b_sel, tt1 , tt2), ak.where(c1, tt1, tt2)),3,axis=1)

        qq_tt = ak.mask(qq, jj_tt_mask) #select leading pT jet > 20 GeV for ttbar
        chi1_tt, mean1_tt, std1_tt = chi_square(tt.t1,194.93 , 47.59 ) #leptonic top
        chi2_tt, mean2_tt, std2_tt = chi_square(tt.t2, 171.55, 44.95 ) #hadronic top
        chi3_tt, mean3_tt, std3_tt = chi_square(qq_tt.mass,73.9, 23.56) #hadronic W
        
        chi_sq_tt = np.sqrt(chi1_tt + chi2_tt + chi3_tt)
        min_chi_sq_tt = ak.argmin(chi_sq_tt, axis=1, keepdims = True) #get index of the minimum chi square 
        chi_sq_tt = chi_sq_tt[min_chi_sq_tt]

        jj_sel_gen_mass_tt =  ak.fill_none(ak.firsts(jj_gen_mass[min_chi_sq_tt]),-1) #gen mass of the di jet pair with minimum chi square
        signal_region_boolean = ak.where(jj_sel_gen_mass_tt > 55.0, 1,
                                         ak.where(jj_sel_gen_mass_tt > 0, 0, 5))

        if self.isMC:
                
            gen = events.GenPart

            gen['isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            nlo = ak.ones_like(events.MET.pt, dtype=float)
            if('TT' in self.dataset): 
                nlo = np.sqrt(corrections.get_ttbar_weight(genTops[:,0].pt) * corrections.get_ttbar_weight(genTops[:,1].pt))
                
            gen['isW'] = (abs(gen.pdgId)==24)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isZ'] = (abs(gen.pdgId)==23)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            
            genWs = gen[gen.isW] 
            genZs = gen[gen.isZ]
            genDYs = gen[gen.isZ&(gen.mass>30)]
            
            nnlo_nlo = {}
            nlo_qcd = ak.ones_like(events.MET.pt, dtype=float)
            nlo_ewk = ak.ones_like(events.MET.pt, dtype=float)
                                            

            ###
            # Isolation weights for muons
            ###

            if hasattr(events, "L1PreFiringWeight"): 
                weights.add('prefiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
            weights.add('genw',events.genWeight)
            weights.add('nlo_ewk',nlo_ewk)
            #weights.add('nlo',nlo) 
            if 'cen' in nnlo_nlo:
                #weights.add('nnlo_nlo',nnlo_nlo['cen'])
                weights.add('qcd1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
                weights.add('qcd2',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
                weights.add('qcd3',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
                weights.add('ew1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
                weights.add('ew2G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
                weights.add('ew3G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
                weights.add('ew2W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
                weights.add('ew3W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
                weights.add('ew2Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
                weights.add('ew3Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
                weights.add('mix',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
                #weights.add('muF',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
                #weights.add('muR',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
            
        lumimask = ak.ones_like(events.MET.pt, dtype=bool) #using events.MET.pt to get 1d array with len(events)
        #if isData:
            #lumimask = lumiMasks[year](events.run, events.luminosityBlock)
        selection.add('lumimask', lumimask)

        # met_filters =  ak.ones_like(events.MET.pt, dtype=bool)
        # #if isData: met_filters = met_filters & events.Flag['eeBadScFilter'] #this filter is recommended for data only
        # for flag in met_filters_names[year]:
        #     met_filters = met_filters & events.Flag[flag]
        # selection.add('met_filters',met_filters)

        # triggers = dak.zeros_like(events.MET.pt, dtype=bool)
        # for trigger_path in singleelectron_triggers[year]:
        #     if not hasattr(events.HLT, trigger_path): continue
        #     triggers = triggers | events.HLT[trigger_path]
        # selection.add('singleelectron_triggers', triggers)
        
        # triggers = dak.zeros_like(events.MET.pt, dtype=bool)
        # for trigger_path in singlemuon_triggers[year]:
        #     if not hasattr(events.HLT, trigger_path): continue
        #     triggers = triggers | events.HLT[trigger_path]
        # selection.add('singlemuon_triggers', triggers)

        noHEMj = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMj = (j_nHEM==0)
        noHEMmet = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMmet = (met.pt>470)|(met.phi>-0.62)|(met.phi<-1.62)    
        
        selection.add('isoneE', (e_ntight==1) & (mu_nloose==0) & (pho_nloose==0) & (tau_nloose==0))
        selection.add('isoneM', (mu_ntight==1) & (e_nloose==0) & (pho_nloose==0) & (tau_nloose==0))
        selection.add('njets',  (j_nsoft>2))
        selection.add('noHEMj', noHEMj)
        selection.add('noHEMmet', noHEMmet)

        # regions = {
        #     'esr': ['isoneE', 'noHEMj', 'njets', 'met_filters', 'noHEMmet'],
        #     'msr': ['isoneM', 'noHEMj', 'njets', 'met_filters', 'noHEMmet']
        #     }
        
        # def normalize(val,cut):
        #     if cut is None:
        #         return ak.fill_none(val, np.nan)
        #     else:
        #         return ak.fill_none(val[cut], np.nan)

        
        # def fill(systematic):
        #     cut = selection.all(*regions[region])
        #     if systematic in weights.variations:
        #         weight = weights.weight(modifier=systematic)[cut]
        #     else:
        #         weight = weights.weight()[cut]
        #     sname = 'nominal' if systematic is None else systematic
        #     variables = {
        #         'met':                         met.pt,
        #         'chi_hadW':                    ak.firsts(chi_sq_hadW),
        #         'chi_hadWs':                   ak.firsts(chi_sq_hadWs),
        #         'chi_tt':                      ak.firsts(chi_sq_tt),
        #     }

        #     for variable in output['hists']:
        #         if variable not in variables:
        #             continue
        #         normalized_variable = {variable: normalize(variables[variable],cut)}
        #         output['hists'][variable].fill(
        #             dataset = self.dataset,
        #             systematic = sname,
        #             signal_region = signal_region_boolean[cut],
        #             **normalized_variable,
        #             weight= weight
        #         )


        #     if systematic is None and self.isMC:
        #         output['nEvents'] = {
        #             'events': len(events),
        #             'genWeights': ak.sum(events.genWeight),
        #             'weights': ak.sum(weights.weight())
        #         }

        #         wgtcutflow = selection.cutflow(*regions[region], weights = weights) #, weightsmodifier = systematic) #weights  
        #         wgtcutflow_result = wgtcutflow.result()
        #         output['cutflow'] = {
        #             'selections': wgtcutflow_result.labels,
        #             'events': wgtcutflow_result.nevcutflow,
        #             'events_min_one': wgtcutflow_result.nevonecut,
        #             'weights': wgtcutflow_result.wgtevcutflow,
        #             'weights_min_one': wgtcutflow_result.wgtevonecut
        #         }
        #         ### AGE: maybe we can save the histogram from wgtcutflow.yieldhist in hists

        # fill(shift_name)

        return output
    
    def postprocess(self, accumulator):
        return accumulator