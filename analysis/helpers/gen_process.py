import awkward as ak
import numpy as np

#def gen_studies(events):
    ######### add gen particles studies here if required #############

    ##gen = events.GenPart
    ##gen['isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
    ##genTops = gen[gen.isTop]
    ##nlo = ak.ones_like(events.MET.pt, dtype=float)
    ##if('TT' in self.dataset): 
    ##    nlo = np.sqrt(corrections.get_ttbar_weight(genTops[:,0].pt) * corrections.get_ttbar_weight(genTops[:,1].pt))
    ##    
    ##gen['isW'] = (abs(gen.pdgId)==24)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
    ##gen['isZ'] = (abs(gen.pdgId)==23)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
    ##
    ##genWs = gen[gen.isW] 
    ##genZs = gen[gen.isZ]
    ##genDYs = gen[gen.isZ&(gen.mass>30)]
    #return events

def gen_process(events, weights):

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
    return weights