import json
import requests

def get_name(ndc):
    try:
        r = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui.json?idtype=NDC&id={ndc}", timeout=2).json()
        if 'rxnormId' not in r.get('idGroup', {}):
            return None
        rxcui = r['idGroup']['rxnormId'][0]
        n = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json", timeout=2).json()
        return n['properties']['name']
    except:
        return None

print("Loading transition graph...")
with open("markov_transitions.json") as f:
    d = json.load(f)

glob = d['global']
diab = d['diabetes']

good_drugs = []
print("Searching for resolvable NDCs with pathway diversions...")
for k in diab.keys():
    if k in glob and len(glob[k]) > 3 and len(diab[k]) > 3:
        top_glob = glob[k][0][0]
        top_diab = diab[k][0][0]
        # We want the top recommendation to be different!
        if top_glob != top_diab:
            name = get_name(k)
            if name is not None:
                good_drugs.append({
                    'ndc': k, 
                    'name': name, 
                    'top_g': get_name(top_glob) or top_glob, 
                    'top_d': get_name(top_diab) or top_diab
                })
                print(f"Success: Found {name} (NDC: {k})")
            if len(good_drugs) >= 3:
                break

print("\n=== BEST DRUG DEMO CODES ===")
for g in good_drugs:
    print(f"Drug: {g['name']} (NDC: {g['ndc']})")
    print(f"  > Global Next:   {g['top_g']}")
    print(f"  > Diabetes Next: {g['top_d']}\n")
