from __future__ import annotations
import json, random, re, hashlib, zipfile, shutil
from pathlib import Path
from collections import Counter, defaultdict

SEED=20260820
ROOT=Path('/mnt/data/semantic_linker_v3_global_stress_dataset')
if ROOT.exists(): shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)

V1_TRAIN=Path('/mnt/data/archive_v1/semantic_linker_train_v1_1.jsonl')

FIELDS=[
'requested_usable_capacity_tib','client_count','average_file_size_gb','max_file_size_gb',
'total_file_count','read_write_ratio','target_read_gbps','target_write_gbps','max_budget_usd',
'max_power_w','annual_growth_percent','__UNRESOLVED__']

DIM={
'requested_usable_capacity_tib':'capacity','client_count':'unknown','average_file_size_gb':'file_size',
'max_file_size_gb':'file_size','total_file_count':'unknown','read_write_ratio':'percent',
'target_read_gbps':'throughput','target_write_gbps':'throughput','max_budget_usd':'money',
'max_power_w':'power','annual_growth_percent':'percent'}

ROLES={
'requested_usable_capacity_tib':['target','current_value','unspecified'],
'client_count':['total_count','unspecified'],
'average_file_size_gb':['average_value','unspecified'],
'max_file_size_gb':['maximum_limit','unspecified'],
'total_file_count':['total_count','unspecified'],
'read_write_ratio':['ratio_component','unspecified'],
'target_read_gbps':['target','minimum_limit','current_value','unspecified'],
'target_write_gbps':['target','minimum_limit','current_value','unspecified'],
'max_budget_usd':['maximum_limit','current_value','unspecified'],
'max_power_w':['maximum_limit','current_value','expected_value','unspecified'],
'annual_growth_percent':['growth_rate','unspecified'],
'__UNRESOLVED__':['unspecified']}

CATEGORIES=[
'clean_natural','paraphrase_synonym','semantic_anchor_typo','mixed_code_switch',
'noisy_mixed_typos','hpc_hard_negative','multi_quantity_collision','alternative_pair',
'complex_role_temporal','current_target_collision','long_context','short_telegraphic',
'punctuation_case_noise','quantity_words','unit_surface_variation','conversational_human_like']

# 500 per category per field -> 8k/field -> 96k stress train; 80 -> 1,280/field -> 15,360 val/test.
COUNT={'train':500,'val':80,'test':80}

VALUES={
'requested_usable_capacity_tib':[80,120,160,200,240,320,400,500,640,720,800,1000,1200,1500,2000],
'client_count':[24,32,48,64,80,96,120,160,200,240,260,275,300,320,400,500,640,800,1000,1200],
'average_file_size_gb':[0.5,1,2,4,8,12,16,24,32,48,64],
'max_file_size_gb':[10,20,40,64,80,100,128,160,200,256,512,1024],
'total_file_count':[100000,500000,1000000,2000000,5000000,8000000,10000000,20000000,50000000,100000000],
'read_write_ratio':[20,25,30,40,50,60,70,75,80,90],
'target_read_gbps':[10,20,30,40,50,60,70,80,88,100,120,160,200],
'target_write_gbps':[5,10,20,30,35,40,50,60,80,100,120],
'max_budget_usd':[50000,75000,80000,100000,120000,150000,180000,200000,250000,300000,500000],
'max_power_w':[5000,8000,10000,12000,15000,18000,20000,25000,30000,40000],
'annual_growth_percent':[5,10,15,20,25,30,35,40,50,60]}

BASE_UNITS={
'requested_usable_capacity_tib':['TiB','TB'],
'average_file_size_gb':['GB','GiB'], 'max_file_size_gb':['GB','GiB'],
'target_read_gbps':['GB/s'], 'target_write_gbps':['GB/s'],
'max_budget_usd':['USD'], 'max_power_w':['W','kW'],
'annual_growth_percent':['%'], 'read_write_ratio':['%'],
'client_count':[None], 'total_file_count':[None]}

ANCHORS={
'requested_usable_capacity_tib':{'en':['usable capacity','storage capacity','usable storage','filesystem capacity','Lustre capacity'], 'fr':['capacité utile','capacité de stockage','stockage utile','capacité Lustre'], 'mixed':['usable capacité','capacité storage','Lustre capacité utile']},
'client_count':{'en':['clients','compute nodes','hosts','endpoints','machines','worker nodes'], 'fr':['clients','nœuds de calcul','noeuds de calcul','hôtes','machines','points de terminaison'], 'mixed':['compute noeuds','noeuds compute','client nodes','compute hôtes','endpoints applicatifs']},
'average_file_size_gb':{'en':['average file size','typical file size','mean file size','representative file size'], 'fr':['taille moyenne des fichiers','taille typique des fichiers','taille moyenne de fichier'], 'mixed':['average taille fichier','taille typical des fichiers']},
'max_file_size_gb':{'en':['maximum file size','largest file size','file size ceiling','max file size'], 'fr':['taille maximale des fichiers','plus gros fichier','taille max de fichier'], 'mixed':['maximum taille fichier','taille max file']},
'total_file_count':{'en':['total file count','number of files','namespace file count','file population','objects in the namespace'], 'fr':['nombre total de fichiers','nombre de fichiers','population de fichiers','objets du namespace'], 'mixed':['total fichiers','file count du namespace']},
'read_write_ratio':{'en':['read share','read ratio','read/write mix','read percentage','I/O read fraction'], 'fr':['part de lecture','ratio lecture','mix lecture/écriture','pourcentage de lecture'], 'mixed':['read part','ratio lecture/write']},
'target_read_gbps':{'en':['read throughput','read bandwidth','target read rate','Lustre read performance'], 'fr':['débit de lecture','bande passante en lecture','performance de lecture','débit lecture cible'], 'mixed':['read débit','débit read Lustre']},
'target_write_gbps':{'en':['write throughput','write bandwidth','target write rate','Lustre write performance'], 'fr':['débit d’écriture','débit d\'écriture','bande passante en écriture','performance d’écriture','débit écriture cible'], 'mixed':['write débit','débit write Lustre']},
'max_budget_usd':{'en':['maximum budget','budget ceiling','spending limit','storage budget'], 'fr':['budget maximal','plafond budgétaire','limite de budget','budget stockage'], 'mixed':['budget ceiling stockage','maximum budget stockage']},
'max_power_w':{'en':['maximum power','power ceiling','power limit','power budget'], 'fr':['puissance maximale','plafond de puissance','limite de puissance','budget de puissance'], 'mixed':['power limite','puissance ceiling']},
'annual_growth_percent':{'en':['annual growth','yearly growth','growth rate','annual capacity growth'], 'fr':['croissance annuelle','taux de croissance','croissance par an'], 'mixed':['annual croissance','growth annuel']}}

# Split-specific mutation operators, designed to avoid memorizing exact typo forms.
def typo(word:str, split:str, rng:random.Random)->str:
    w=word
    if len(w)<5: return w+'x'
    letters=list(w)
    idxs=[i for i,c in enumerate(letters) if c.isalpha() and 0<i<len(letters)-1]
    if not idxs: return w+'x'
    i=rng.choice(idxs)
    if split=='train':
        if rng.random()<0.5:
            del letters[i]
        else:
            j=min(i+1,len(letters)-1); letters[i],letters[j]=letters[j],letters[i]
    elif split=='val':
        if rng.random()<0.5:
            letters.insert(i,letters[i])
        else:
            repl='aeiou'[rng.randrange(5)]
            letters[i]=repl if letters[i].lower()!=repl else 'x'
    else:
        if rng.random()<0.5:
            letters.insert(i,'x' if letters[i].lower()!='x' else 'z')
        else:
            j=min(i+2,len(letters)-1)
            a=letters[i:j+1]
            if len(a)>=3: a=a[1:]+a[:1]
            letters[i:j+1]=a
    return ''.join(letters)

ONES_EN=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
TENS_EN=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
ONES_FR=['zéro','un','deux','trois','quatre','cinq','six','sept','huit','neuf','dix','onze','douze','treize','quatorze','quinze','seize','dix-sept','dix-huit','dix-neuf']
TENS_FR={20:'vingt',30:'trente',40:'quarante',50:'cinquante',60:'soixante'}

def words_en(n):
    if isinstance(n,float) and not n.is_integer():
        a,b=str(n).split('.'); return words_en(int(a))+' point '+' '.join(ONES_EN[int(d)] for d in b)
    n=int(n)
    if n<20: return ONES_EN[n]
    if n<100:
        return TENS_EN[n//10] + ('' if n%10==0 else '-'+ONES_EN[n%10])
    if n<1000:
        return ONES_EN[n//100]+' hundred'+('' if n%100==0 else ' '+words_en(n%100))
    if n<1_000_000:
        return words_en(n//1000)+' thousand'+('' if n%1000==0 else ' '+words_en(n%1000))
    if n<1_000_000_000:
        return words_en(n//1_000_000)+' million'+('' if n%1_000_000==0 else ' '+words_en(n%1_000_000))
    return str(n)

def words_fr_under100(n):
    if n<20: return ONES_FR[n]
    if n<70:
        t=(n//10)*10; u=n%10; base=TENS_FR[t]
        if u==0: return base
        if u==1: return base+' et un'
        return base+'-'+ONES_FR[u]
    if n<80:
        return 'soixante-'+ONES_FR[n-60] if n!=71 else 'soixante et onze'
    if n==80: return 'quatre-vingts'
    rem=n-80
    return 'quatre-vingt-'+ONES_FR[rem]

def words_fr(n):
    if isinstance(n,float) and not n.is_integer():
        a,b=str(n).split('.'); return words_fr(int(a))+' virgule '+' '.join(ONES_FR[int(d)] for d in b)
    n=int(n)
    if n<100: return words_fr_under100(n)
    if n<1000:
        h=n//100; rem=n%100
        base='cent' if h==1 else ONES_FR[h]+' cent'+('s' if rem==0 else '')
        return base if rem==0 else base+' '+words_fr_under100(rem)
    if n<1_000_000:
        th=n//1000; rem=n%1000
        base='mille' if th==1 else words_fr(th)+' mille'
        return base if rem==0 else base+' '+words_fr(rem)
    if n<1_000_000_000:
        m=n//1_000_000; rem=n%1_000_000
        base=('un million' if m==1 else words_fr(m)+' millions')
        return base if rem==0 else base+' '+words_fr(rem)
    return str(n)

def typo_number_phrase(raw, split, rng):
    # Corrupt one alphabetic token while preserving the unit token when possible.
    toks=raw.split()
    candidates=[i for i,t in enumerate(toks) if len(re.sub(r'[^A-Za-zÀ-ÿ]','',t))>=5 and t.lower() not in {'percent','dollars','kilowatts','tebibytes','gigabytes'}]
    if not candidates: return raw
    i=rng.choice(candidates); toks[i]=typo(toks[i],split,rng); return ' '.join(toks)

def quantity_words(field, value, lang, rng, split):
    conv=words_fr if lang=='fr' else words_en
    unit=rng.choice(BASE_UNITS[field])
    source_value=value
    if field=='max_power_w':
        # Prefer compact source-unit wording: 15000 W -> fifteen kilowatts / quinze kilowatts.
        if value>=1000 and value%1000==0:
            source_value=value//1000; unit='kW'; raw=conv(source_value)+(' kilowatts' if lang!='fr' else ' kilowatts')
        else:
            unit='W'; raw=conv(value)+(' watts')
    elif field=='max_budget_usd':
        unit='USD'; raw=conv(value)+(' dollars')
    elif field in ('annual_growth_percent','read_write_ratio'):
        unit='%'; raw=conv(value)+(' pour cent' if lang=='fr' else ' percent')
    else:
        if field in ('client_count','total_file_count'): unit=None
        raw=conv(value)
        if unit: raw+=' '+unit
    corrected=False
    if rng.random()<0.5:
        noisy=typo_number_phrase(raw,split,rng)
        corrected=(noisy!=raw); raw=noisy
    return raw,source_value,unit,corrected

def role_for(field, rng):
    allowed=ROLES[field]
    # favor informative roles over unspecified
    weighted=[]
    for r in allowed:
        weighted.extend([r]*(1 if r=='unspecified' else 5))
    return rng.choice(weighted)

def qspec(field, value, rng, category='clean_natural'):
    unit=rng.choice(BASE_UNITS[field])
    display_value=value
    if field=='max_power_w' and unit=='kW':
        # Quantity.value follows the scanner contract: numeric value in the SOURCE unit.
        display_value=value/1000
        if float(display_value).is_integer(): display_value=int(display_value)
    if field in ('client_count','total_file_count'):
        unit=None
    if isinstance(display_value,float) and display_value.is_integer(): display_value=int(display_value)
    raw=f'{display_value}' if unit is None else f'{display_value} {unit}'
    source_value=display_value
    if field=='max_budget_usd' and category=='unit_surface_variation':
        raw=rng.choice([f'${int(value):,}',f'{int(value)} dollars',f'USD {int(value)}'])
        unit='USD'; source_value=value
    elif field=='max_power_w' and category=='unit_surface_variation':
        kw=value/1000; kw=int(kw) if float(kw).is_integer() else kw
        raw=rng.choice([f'{kw} kW',f'{kw}kW',f'{kw} kilowatts'])
        unit='kW'; source_value=kw
    elif field in ('target_read_gbps','target_write_gbps') and category=='unit_surface_variation':
        raw=rng.choice([f'{value} GB/s',f'{value} GBps',f'{value} GB per second'])
        unit='GB/s'; source_value=value
    elif field=='requested_usable_capacity_tib' and category=='unit_surface_variation':
        raw=rng.choice([f'{value} TiB',f'{value} tebibytes',f'{value} TB'])
        unit='TiB' if ('TiB' in raw or 'tebibytes' in raw) else 'TB'; source_value=value
    elif field in ('average_file_size_gb','max_file_size_gb') and category=='unit_surface_variation':
        raw=rng.choice([f'{value} GB',f'{value} gigabytes',f'{value} GiB'])
        unit='GB' if ('GB' in raw or 'gigabytes' in raw) else 'GiB'; source_value=value
    elif field in ('annual_growth_percent','read_write_ratio') and category=='unit_surface_variation':
        raw=rng.choice([f'{value}%',f'{value} percent',f'{value} pour cent'])
        unit='%'; source_value=value
    return raw, source_value, unit

def normalized_quantity(value, unit):
    if isinstance(value,float) and value.is_integer(): value=int(value)
    return str(value) if unit is None else f'{value} {unit}'

def field_phrase(field, role, lang, anchor, Q, rng, split, category):
    # Hand-authored phrase banks with neutral grammar around variable anchors.
    if lang=='en':
        if field=='requested_usable_capacity_tib':
            if role=='current_value': return rng.choice([f'Current {anchor}: {Q}.',f'We currently operate with {Q} of usable Lustre storage.'])
            return rng.choice([f'We need about {Q} of {anchor} for production.',f'Target {anchor}: {Q}.',f'Please size the filesystem for {Q} of usable capacity.'])
        if field=='client_count': return rng.choice([f'Approximately {Q} {anchor} will mount Lustre.',f'We expect {Q} {anchor} to access the shared filesystem.',f'The production namespace must serve {Q} {anchor}.'])
        if field=='average_file_size_gb': return rng.choice([f'For {anchor}, the expected value is around {Q}.',f'A representative user file averages {Q}.',f'Files are typically about {Q} each.'])
        if field=='max_file_size_gb': return rng.choice([f'For {anchor}, the limit is {Q}.',f'We need to support individual files up to {Q}.',f'The largest expected file is {Q}.'])
        if field=='total_file_count': return rng.choice([f'For {anchor}, the expected total is {Q}.',f'We expect roughly {Q} files in the Lustre namespace.',f'Production should support {Q} filesystem objects.'])
        if field=='read_write_ratio': return rng.choice([f'Reads represent about {Q} of the I/O mix.',f'The workload is roughly {Q} reads, with the rest being writes.',f'For {anchor}, the expected value is {Q}.'])
        if field=='target_read_gbps':
            if role=='current_value': return f'Current value for {anchor}: {Q}.'
            if role=='minimum_limit': return f'Read performance must sustain at least {Q}.'
            return rng.choice([f'Target value for {anchor}: {Q}.',f'We want Lustre reads to reach about {Q}.'])
        if field=='target_write_gbps':
            if role=='current_value': return f'Current value for {anchor}: {Q}.'
            if role=='minimum_limit': return f'Write performance must sustain at least {Q}.'
            return rng.choice([f'Target value for {anchor}: {Q}.',f'We want Lustre writes to reach about {Q}.'])
        if field=='max_budget_usd':
            if role=='current_value': return f'Current value for {anchor}: {Q}.'
            return rng.choice([f'Limit for {anchor}: {Q}.',f'We cannot spend more than {Q} on the storage architecture.',f'Keep the total storage cost below {Q}.'])
        if field=='max_power_w':
            if role=='current_value': return f'Current measured value for {anchor}: {Q}.'
            if role=='expected_value': return f'Expected value for {anchor}: {Q}.'
            return rng.choice([f'Limit for {anchor}: {Q}.',f'Power consumption must stay below {Q}.',f'Do not exceed {Q} for the storage system.'])
        if field=='annual_growth_percent': return rng.choice([f'Expected value for {anchor}: {Q}.',f'Capacity should grow by about {Q} each year.',f'Plan for an annual data growth rate of {Q}.'])
    elif lang=='fr':
        if field=='requested_usable_capacity_tib':
            if role=='current_value': return rng.choice([f'Valeur actuelle pour {anchor} : {Q}.',f'Nous disposons actuellement de {Q} de stockage Lustre utile.'])
            return rng.choice([f'Besoin de production pour {anchor} : environ {Q}.',f'Cible pour {anchor} : {Q}.',f'Il faut dimensionner le système pour {Q} de capacité utile.'])
        if field=='client_count': return rng.choice([f'Environ {Q} {anchor} monteront Lustre.',f'Nous prévoyons {Q} {anchor} qui accéderont au système de fichiers partagé.',f'Le namespace de production devra servir {Q} {anchor}.'])
        if field=='average_file_size_gb': return rng.choice([f'Valeur attendue pour {anchor} : environ {Q}.',f'Un fichier utilisateur représentatif fait en moyenne {Q}.',f'Les fichiers font typiquement {Q}.'])
        if field=='max_file_size_gb': return rng.choice([f'Limite associée à {anchor} : {Q}.',f'Il faut supporter des fichiers individuels jusqu’à {Q}.',f'Le plus gros fichier attendu fait {Q}.'])
        if field=='total_file_count': return rng.choice([f'Total attendu pour {anchor} : {Q}.',f'Nous prévoyons environ {Q} fichiers dans le namespace Lustre.',f'La production doit supporter {Q} objets de système de fichiers.'])
        if field=='read_write_ratio': return rng.choice([f'Les lectures représentent environ {Q} du mix I/O.',f'La charge est composée d’environ {Q} de lectures, le reste étant des écritures.',f'Valeur attendue pour {anchor} : {Q}.'])
        if field=='target_read_gbps':
            if role=='current_value': return f'Valeur actuelle pour {anchor} : {Q}.'
            if role=='minimum_limit': return f'La lecture doit soutenir au moins {Q}.'
            return rng.choice([f'Cible pour {anchor} : {Q}.',f'Nous visons environ {Q} en lecture Lustre.'])
        if field=='target_write_gbps':
            if role=='current_value': return f'Valeur actuelle pour {anchor} : {Q}.'
            if role=='minimum_limit': return f'L’écriture doit soutenir au moins {Q}.'
            return rng.choice([f'Cible pour {anchor} : {Q}.',f'Nous visons environ {Q} en écriture Lustre.'])
        if field=='max_budget_usd':
            if role=='current_value': return f'Valeur actuelle pour {anchor} : {Q}.'
            return rng.choice([f'Limite pour {anchor} : {Q}.',f'Nous ne pouvons pas dépasser {Q} pour l’architecture de stockage.',f'Le coût total du stockage doit rester sous {Q}.'])
        if field=='max_power_w':
            if role=='current_value': return f'Valeur mesurée actuellement pour {anchor} : {Q}.'
            if role=='expected_value': return f'Valeur attendue pour {anchor} : environ {Q}.'
            return rng.choice([f'Limite pour {anchor} : {Q}.',f'La consommation doit rester sous {Q}.',f'Le système de stockage ne doit pas dépasser {Q}.'])
        if field=='annual_growth_percent': return rng.choice([f'Valeur attendue pour {anchor} : {Q}.',f'La capacité devrait augmenter d’environ {Q} par an.',f'Il faut prévoir un taux de croissance annuel de {Q}.'])
    # Mixed EN/FR: code-switching by complete semantic chunks, not broken grammar.
    if field=='client_count': return rng.choice([f'Approximately {Q} compute nodes vont monter Lustre.',f'On prévoit {Q} compute nodes pour access le filesystem Lustre.',f'Production namespace: {Q} clients HPC à servir.'])
    if field=='requested_usable_capacity_tib':
        if role=='current_value': return rng.choice([f'Current usable capacity pour Lustre : {Q}.',f'Storage utile actuel : {Q}.'])
        return rng.choice([f'Target usable capacity pour Lustre : {Q}.',f'On vise une usable capacity de {Q} pour la production.',f'Storage utile target : {Q}.'])
    if field=='average_file_size_gb': return rng.choice([f'Average file size estimée : {Q}.',f'Pour les fichiers typical, on est autour de {Q}.'])
    if field=='max_file_size_gb': return rng.choice([f'Max file size autorisée : {Q}.',f'On doit supporter des fichiers up to {Q}.'])
    if field=='total_file_count': return rng.choice([f'Namespace file count prévu : {Q}.',f'On prévoit {Q} filesystem objects en production.'])
    if field=='read_write_ratio': return rng.choice([f'Read share prévue : {Q}.',f'Le workload est environ {Q} read, le reste write.'])
    if field=='target_read_gbps':
        if role=='current_value': return f'Current read throughput pour Lustre : {Q}.'
        if role=='minimum_limit': return f'Minimum read throughput requis : {Q}.'
        return rng.choice([f'Target read throughput pour Lustre : {Q}.',f'On vise un read throughput de {Q}.'])
    if field=='target_write_gbps':
        if role=='current_value': return f'Current write throughput pour Lustre : {Q}.'
        if role=='minimum_limit': return f'Minimum write throughput requis : {Q}.'
        return rng.choice([f'Target write throughput pour Lustre : {Q}.',f'On vise un write throughput de {Q}.'])
    if field=='max_budget_usd':
        if role=='current_value': return f'Current storage budget : {Q}.'
        return rng.choice([f'Storage budget ceiling : {Q}.',f'Budget stockage, hard limit : {Q}.'])
    if field=='max_power_w':
        if role=='current_value': return f'Current storage power : {Q}.'
        if role=='expected_value': return f'Expected storage power : {Q}.'
        return rng.choice([f'Storage power limit : {Q}.',f'Puissance storage, hard ceiling : {Q}.'])
    if field=='annual_growth_percent': return rng.choice([f'Annual growth prévue : {Q}.',f'Croissance annuelle target : {Q}.'])
    raise KeyError(field)

def unresolved_example(lang,Q,rng,category,dimension):
    if dimension=='unknown':
        if lang=='en': return f'The compute node has {Q} CPU cores; this is hardware inventory, not a Lustre client requirement.'
        if lang=='fr': return f'Le nœud de calcul possède {Q} cœurs CPU ; il s’agit d’un inventaire matériel, pas d’un nombre de clients Lustre.'
        return f'Compute node hardware: {Q} CPU cores; ce n’est pas un client-count requirement.'
    if dimension=='throughput':
        if lang=='en': return f'An InfiniBand link benchmark reported {Q}, but that value is network performance, not a Lustre throughput requirement.'
        if lang=='fr': return f'Un benchmark InfiniBand a mesuré {Q}, mais cette valeur concerne le réseau, pas un besoin de débit Lustre.'
        return f'InfiniBand benchmark: {Q}; c’est network performance, not a Lustre throughput target.'
    if dimension=='capacity':
        if lang=='en': return f'The archive tier provides {Q}, but that is archive capacity, not the requested Lustre usable capacity.'
        if lang=='fr': return f'Le niveau d’archive fournit {Q}, mais il s’agit de capacité d’archive, pas de la capacité utile Lustre demandée.'
        return f'Archive tier capacity: {Q}; ce n’est pas la requested usable Lustre capacity.'
    if dimension=='power':
        if lang=='en': return f'A rack sensor displayed {Q} during a stress test; this is a measurement, not the storage power requirement.'
        if lang=='fr': return f'Un capteur de rack a affiché {Q} pendant un stress test ; c’est une mesure, pas la limite de puissance du stockage.'
        return f'Rack sensor reading: {Q}; c’est une mesure, not the storage power limit.'
    if dimension=='money':
        if lang=='en': return f'A vendor quote mentions {Q} for an unrelated service; this is not the maximum Lustre storage budget.'
        if lang=='fr': return f'Un devis fournisseur mentionne {Q} pour un service distinct ; ce n’est pas le budget maximal du stockage Lustre.'
        return f'Vendor quote: {Q} pour un service séparé; not the Lustre storage budget.'
    return f'The monitoring note contains {Q}, but it is not a Lustre requirement.'

def human_like_phrase(field, role, lang, Q, rng):
    if lang=='en':
        if field=='requested_usable_capacity_tib': return f'I think we will need roughly {Q} of usable Lustre capacity.' if role!='current_value' else f'Right now we have about {Q} of usable Lustre capacity.'
        if field=='client_count': return f'From what we know today, around {Q} clients should mount Lustre.'
        if field=='average_file_size_gb': return f'Typical user files are probably around {Q}.'
        if field=='max_file_size_gb': return f'Some files can get as large as {Q}, so we need to support that size.'
        if field=='total_file_count': return f'We may end up with roughly {Q} files in the namespace.'
        if field=='read_write_ratio': return f'I would estimate that about {Q} of the I/O will be reads.'
        if field=='target_read_gbps':
            if role=='current_value': return f'At the moment, reads are around {Q}.'
            if role=='minimum_limit': return f'For reads, we really need at least {Q}.'
            return f'For reads, we would like to reach roughly {Q}.'
        if field=='target_write_gbps':
            if role=='current_value': return f'At the moment, writes are around {Q}.'
            if role=='minimum_limit': return f'For writes, we really need at least {Q}.'
            return f'For writes, we would like to reach roughly {Q}.'
        if field=='max_budget_usd': return f'Budget-wise, we cannot go above {Q}.' if role!='current_value' else f'Our current storage budget is about {Q}.'
        if field=='max_power_w':
            if role=='current_value': return f'Right now the storage draws about {Q}.'
            if role=='expected_value': return f'We expect the storage to draw roughly {Q}.'
            return f'For power, we should stay below {Q}.'
        if field=='annual_growth_percent': return f'I would plan for roughly {Q} annual data growth.'
    if lang=='fr':
        if field=='requested_usable_capacity_tib': return f'Je pense qu’il nous faudra environ {Q} de capacité Lustre utile.' if role!='current_value' else f'Actuellement, nous avons environ {Q} de capacité Lustre utile.'
        if field=='client_count': return f'D’après nos estimations, environ {Q} clients devraient monter Lustre.'
        if field=='average_file_size_gb': return f'Les fichiers utilisateurs font probablement autour de {Q} en moyenne.'
        if field=='max_file_size_gb': return f'Certains fichiers peuvent atteindre {Q}, il faut donc supporter cette taille.'
        if field=='total_file_count': return f'On pourrait arriver à environ {Q} fichiers dans le namespace.'
        if field=='read_write_ratio': return f'Je dirais qu’environ {Q} des I/O seront des lectures.'
        if field=='target_read_gbps':
            if role=='current_value': return f'Actuellement, la lecture est autour de {Q}.'
            if role=='minimum_limit': return f'En lecture, il nous faut vraiment au moins {Q}.'
            return f'En lecture, on aimerait atteindre environ {Q}.'
        if field=='target_write_gbps':
            if role=='current_value': return f'Actuellement, l’écriture est autour de {Q}.'
            if role=='minimum_limit': return f'En écriture, il nous faut vraiment au moins {Q}.'
            return f'En écriture, on aimerait atteindre environ {Q}.'
        if field=='max_budget_usd': return f'Côté budget, on ne peut pas dépasser {Q}.' if role!='current_value' else f'Notre budget stockage actuel est d’environ {Q}.'
        if field=='max_power_w':
            if role=='current_value': return f'Actuellement, le stockage consomme environ {Q}.'
            if role=='expected_value': return f'On s’attend à une consommation stockage d’environ {Q}.'
            return f'Pour la puissance, il faudrait rester sous {Q}.'
        if field=='annual_growth_percent': return f'Je prévoirais environ {Q} de croissance des données par an.'
    # mixed, natural code switching by chunks
    if field=='requested_usable_capacity_tib': return f'Pour notre use case, usable Lustre capacity autour de {Q} devrait suffire.'
    if field=='client_count': return f'On devrait avoir around {Q} Lustre clients en production.'
    if field=='average_file_size_gb': return f'Typical file size chez nous : autour de {Q}.'
    if field=='max_file_size_gb': return f'Largest files peuvent atteindre {Q}, donc il faut les supporter.'
    if field=='total_file_count': return f'On peut finir avec around {Q} files dans le namespace.'
    if field=='read_write_ratio': return f'Je dirais around {Q} read dans le workload.'
    if field=='target_read_gbps':
        if role=='current_value': return f'Current read performance : environ {Q}.'
        if role=='minimum_limit': return f'Read performance : il nous faut at least {Q}.'
        return f'Read target pour Lustre : environ {Q}.'
    if field=='target_write_gbps':
        if role=='current_value': return f'Current write performance : environ {Q}.'
        if role=='minimum_limit': return f'Write performance : il nous faut at least {Q}.'
        return f'Write target pour Lustre : environ {Q}.'
    if field=='max_budget_usd': return f'Budget storage : on ne peut pas dépasser {Q}.'
    if field=='max_power_w':
        if role=='current_value': return f'Current storage power : environ {Q}.'
        if role=='expected_value': return f'Expected storage power : environ {Q}.'
        return f'Power limit storage : rester sous {Q}.'
    if field=='annual_growth_percent': return f'Annual data growth prévue : autour de {Q}.'
    raise KeyError(field)

def confounder(lang,rng):
    if lang=='en': return rng.choice(['64 CPU cores','8 GPUs','4 OSS servers','2 metadata servers','100 Gb/s network link','32 MPI ranks','16 OSTs','512 GB of RAM'])
    if lang=='fr': return rng.choice(['64 cœurs CPU','8 GPU','4 serveurs OSS','2 serveurs de métadonnées','un lien réseau à 100 Gb/s','32 rangs MPI','16 OST','512 GB de RAM'])
    return rng.choice(['64 CPU cores','8 GPU compute','4 serveurs OSS','100 Gb/s network link','32 MPI ranks'])

def other_field_quantity(field,rng):
    others=[f for f in FIELDS if f not in (field,'__UNRESOLVED__')]
    of=rng.choice(others); v=rng.choice(VALUES[of]); raw,val,unit=qspec(of,v,rng)
    return of,raw

def mark_from_template(template, raw):
    assert template.count('{Q}')==1
    original=template.replace('{Q}',raw)
    marked=template.replace('{Q}',f'[Q]{raw}[/Q]')
    return original,marked

def make_regular(field,category,split,idx,lang,rng):
    if field=='__UNRESOLVED__':
        # choose realistic distractor dimensions
        unresolved_choices=[('unknown',str(rng.choice([8,16,32,64,128,256])),None),('throughput',f"{rng.choice([25,50,100,200])} GB/s",'GB/s'),('capacity',f"{rng.choice([100,200,500,800])} TiB",'TiB'),('power',f"{rng.choice([8,12,15,20])} kW",'kW'),('money',f"{rng.choice([50000,100000,150000])} USD",'USD')]
        dim,raw,unit=rng.choice(unresolved_choices); val=float(re.findall(r'[0-9.]+',raw.replace(',',''))[0]); val=int(val) if val.is_integer() else val
        template=unresolved_example(lang,'{Q}',rng,category,dim)
        original,marked=mark_from_template(template,raw)
        return original,marked,raw,val,unit,dim,'unspecified',False
    role=role_for(field,rng)
    value=rng.choice(VALUES[field])
    if category=='quantity_words': raw,val,unit,number_corrected=quantity_words(field,value,lang,rng,split)
    else:
        raw,val,unit=qspec(field,value,rng,category); number_corrected=False
    anchor_lang=lang if lang in ('en','fr','mixed') else 'en'
    anchor=rng.choice(ANCHORS[field][anchor_lang])
    if category in ('semantic_anchor_typo','noisy_mixed_typos'):
        # corrupt one anchor token >4 chars using split-specific operator
        toks=anchor.split(); eligible=[i for i,t in enumerate(toks) if len(re.sub(r'[^A-Za-zÀ-ÿ]','',t))>=5]
        if eligible:
            j=rng.choice(eligible); toks[j]=typo(toks[j],split,rng); anchor=' '.join(toks)
    template=field_phrase(field,role,lang,anchor,'{Q}',rng,split,category)
    if category in ('semantic_anchor_typo','noisy_mixed_typos') and anchor not in template:
        prefix = (f'Requirement concept: {anchor}. ' if lang=='en' else (f'Concept du besoin : {anchor}. ' if lang=='fr' else f'Requirement concept : {anchor}. '))
        template=prefix+template
    if category=='paraphrase_synonym':
        if lang=='en': template=rng.choice(['From the sizing perspective, ','For production, ','As a planning assumption, '])+template
        elif lang=='fr': template=rng.choice(['Pour le dimensionnement, ','En production, ','Comme hypothèse de capacité, '])+template
        else: template=rng.choice(['For sizing, ','En production, '])+template
    elif category=='long_context':
        pre={'en':'We are preparing the next production Lustre deployment. The compute environment already exists and the storage team is validating the final sizing assumptions. ',
             'fr':'Nous préparons le prochain déploiement Lustre de production. L’environnement de calcul existe déjà et l’équipe stockage valide les hypothèses finales de dimensionnement. ',
             'mixed':'We are preparing le prochain Lustre deployment. L’équipe storage valide maintenant les sizing assumptions. '}[lang]
        post={'en':' This figure refers to the shared production filesystem rather than a temporary benchmark.',
              'fr':' Cette valeur concerne le système de fichiers partagé de production et non un benchmark temporaire.',
              'mixed':' Cette valeur concerne the shared production filesystem, pas un benchmark temporaire.'}[lang]
        template=pre+template+post
    elif category=='short_telegraphic':
        template={
            'en':f'{anchor}: {{Q}}.', 'fr':f'{anchor} : {{Q}}.', 'mixed':f'{anchor} target: {{Q}}.'}[lang]
    elif category=='punctuation_case_noise':
        template=template.replace('{Q}','__QTARGET__')
        template=template.upper() if rng.random()<0.5 else template.lower()
        template=template.replace('__qtarget__','__QTARGET__').replace('__QTARGET__','{Q}')
        template=template.replace(' ',rng.choice(['  ',' ','   ']))
        template=rng.choice(['>> ','REQ: ','# '])+template
    elif category=='mixed_code_switch':
        # Force a natural mixed-language construction even if scheduled lang was en/fr.
        lang='mixed'; anchor=rng.choice(ANCHORS[field]['mixed']); template=field_phrase(field,role,'mixed',anchor,'{Q}',rng,split,category)
    elif category=='noisy_mixed_typos':
        # If not already mixed, add light code switching plus semantic typo only; avoid gibberish.
        if lang=='en': template=template.replace('production','production HPC').replace('filesystem','filesystem partagé')
        elif lang=='fr': template=template.replace('production','production HPC').replace('système de fichiers','filesystem')
    elif category=='hpc_hard_negative':
        c=confounder(lang,rng)
        if lang=='en': template=f'The cluster inventory shows {c}, but that is not the sizing quantity here. '+template
        elif lang=='fr': template=f'L’inventaire du cluster indique {c}, mais ce n’est pas la quantité à dimensionner ici. '+template
        else: template=f'Cluster inventory: {c}; ce n’est pas la target ici. '+template
    elif category=='multi_quantity_collision':
        of,oraw=other_field_quantity(field,rng)
        if lang=='en': template=template+f' A separate requirement mentions {oraw} for another storage parameter.'
        elif lang=='fr': template=template+f' Une autre exigence mentionne {oraw} pour un autre paramètre de stockage.'
        else: template=template+f' Another requirement mentionne {oraw} pour un autre field.'
    elif category=='conversational_human_like':
        template=human_like_phrase(field,role,lang,'{Q}',rng)
    # Categories alternative_pair / complex roles / current-target handled elsewhere.
    original,marked=mark_from_template(template,raw)
    corrected=(category in ('semantic_anchor_typo','noisy_mixed_typos')) or number_corrected
    return original,marked,raw,val,unit,DIM[field],role,corrected

def alt_text(field,lang,q1,q2,role,rng):
    a=rng.choice(ANCHORS[field][lang if lang in ('en','fr','mixed') else 'en']) if field!='__UNRESOLVED__' else 'measurement'
    if field=='__UNRESOLVED__':
        if lang=='en': return f'The network diagnostic may show {q1} or {q2}; neither value is a Lustre storage requirement.'
        if lang=='fr': return f'Le diagnostic réseau peut afficher {q1} ou {q2} ; aucune des deux valeurs n’est une exigence Lustre.'
        return f'Network diagnostic : {q1} or {q2}; aucune de ces valeurs n’est une storage requirement.'
    if lang=='en':
        return rng.choice([f'For {a}, the value could be {q1} or {q2}.',f'We are considering either {q1} or {q2} for {a}.',f'For {a}, the two candidate values are {q1} and {q2}.'])
    if lang=='fr':
        return rng.choice([f'Pour {a}, la valeur pourrait être {q1} ou {q2}.',f'Pour {a}, nous envisageons soit {q1}, soit {q2}.',f'Pour {a}, les deux valeurs candidates sont {q1} et {q2}.'])
    return rng.choice([f'Pour {a}, candidate values : {q1} or {q2}.',f'For {a}, on considère soit {q1}, soit {q2}.'])

def role_temporal_pair(field,lang,q1,q2,rng):
    if field=='requested_usable_capacity_tib': roles=('current_value','target')
    elif field in ('target_read_gbps','target_write_gbps'): roles=('current_value','target')
    elif field=='max_budget_usd': roles=('current_value','maximum_limit')
    elif field=='max_power_w': roles=('current_value',rng.choice(['expected_value','maximum_limit']))
    elif field=='__UNRESOLVED__': roles=('unspecified','unspecified')
    else:
        base=[r for r in ROLES[field] if r!='unspecified']; r=base[0] if base else 'unspecified'; roles=(r,r)
    anchor=rng.choice(ANCHORS[field][lang if lang in ('en','fr','mixed') else 'en']) if field!='__UNRESOLVED__' else 'measurement'
    if field=='__UNRESOLVED__':
        if lang=='en': text=f'The lab benchmark was {q1} last week and {q2} today; neither is a production Lustre requirement.'
        elif lang=='fr': text=f'Le benchmark de laboratoire était à {q1} la semaine dernière et à {q2} aujourd’hui ; aucune valeur n’est une exigence Lustre.'
        else: text=f'Lab benchmark : {q1} last week, {q2} aujourd’hui; neither is a production requirement.'
        return text,roles
    multirole = field in ('requested_usable_capacity_tib','target_read_gbps','target_write_gbps','max_budget_usd','max_power_w')
    if multirole:
        if field in ('requested_usable_capacity_tib','target_read_gbps','target_write_gbps'):
            if lang=='en': text=f'For {anchor}, the current value is {q1}, while the production target is {q2}.'
            elif lang=='fr': text=f'Pour {anchor}, la valeur actuelle est {q1}, tandis que la cible de production est {q2}.'
            else: text=f'Pour {anchor}, current value : {q1}; production target : {q2}.'
        elif field=='max_budget_usd':
            if lang=='en': text=f'For {anchor}, the current value is {q1}, while the spending ceiling is {q2}.'
            elif lang=='fr': text=f'Pour {anchor}, la valeur actuelle est {q1}, tandis que le plafond budgétaire est {q2}.'
            else: text=f'Pour {anchor}, current value : {q1}; budget ceiling : {q2}.'
        else: # max_power_w
            if roles[1]=='expected_value':
                if lang=='en': text=f'For {anchor}, the current value is {q1}, while the expected production value is {q2}.'
                elif lang=='fr': text=f'Pour {anchor}, la valeur actuelle est {q1}, tandis que la valeur attendue en production est {q2}.'
                else: text=f'Pour {anchor}, current value : {q1}; expected production value : {q2}.'
            else:
                if lang=='en': text=f'For {anchor}, the current value is {q1}, while the hard power ceiling is {q2}.'
                elif lang=='fr': text=f'Pour {anchor}, la valeur actuelle est {q1}, tandis que le plafond strict de puissance est {q2}.'
                else: text=f'Pour {anchor}, current value : {q1}; hard power ceiling : {q2}.'
    else:
        if lang=='en': text=f'For {anchor}, scenario A gives {q1}, while scenario B gives {q2} for the same requirement type.'
        elif lang=='fr': text=f'Pour {anchor}, le scénario A donne {q1}, tandis que le scénario B donne {q2} pour le même type de besoin.'
        else: text=f'Pour {anchor}, scenario A : {q1}; scenario B : {q2}; same requirement type.'
    return text,roles

def role_collision_pair(field,lang,q1,q2,rng):
    if field=='requested_usable_capacity_tib': roles=('current_value','target')
    elif field in ('target_read_gbps','target_write_gbps'): roles=('minimum_limit','target')
    elif field=='max_budget_usd': roles=('current_value','maximum_limit')
    elif field=='max_power_w': roles=('expected_value','maximum_limit')
    elif field=='__UNRESOLVED__': roles=('unspecified','unspecified')
    else:
        base=[r for r in ROLES[field] if r!='unspecified']; r=base[0] if base else 'unspecified'; roles=(r,r)
    anchor=rng.choice(ANCHORS[field][lang if lang in ('en','fr','mixed') else 'en']) if field!='__UNRESOLVED__' else 'measurement'
    if field=='__UNRESOLVED__':
        if lang=='en': text=f'The hardware sheet lists {q1}, while a lab note lists {q2}; neither is a Lustre requirement.'
        elif lang=='fr': text=f'La fiche matérielle indique {q1}, tandis qu’une note de laboratoire indique {q2} ; aucune valeur n’est une exigence Lustre.'
        else: text=f'Hardware sheet : {q1}; lab note : {q2}; neither is a Lustre requirement.'
        return text,roles
    if field in ('target_read_gbps','target_write_gbps'):
        if lang=='en': text=f'For {anchor}, the minimum acceptable value is {q1}, while the preferred production target is {q2}.'
        elif lang=='fr': text=f'Pour {anchor}, le minimum acceptable est {q1}, tandis que la cible de production préférée est {q2}.'
        else: text=f'Pour {anchor}, minimum acceptable : {q1}; preferred target : {q2}.'
    elif field=='max_power_w':
        if lang=='en': text=f'For {anchor}, the expected value is {q1}, but the hard ceiling remains {q2}.'
        elif lang=='fr': text=f'Pour {anchor}, la valeur attendue est {q1}, mais le plafond strict reste {q2}.'
        else: text=f'Pour {anchor}, expected value : {q1}; hard ceiling : {q2}.'
    elif field=='max_budget_usd':
        if lang=='en': text=f'For {anchor}, the current value is {q1}, and the absolute spending ceiling is {q2}.'
        elif lang=='fr': text=f'Pour {anchor}, la valeur actuelle est {q1}, et le plafond absolu est {q2}.'
        else: text=f'Pour {anchor}, current value : {q1}; absolute ceiling : {q2}.'
    elif field=='requested_usable_capacity_tib':
        if lang=='en': text=f'For {anchor}, the current value is {q1}; the new production target is {q2}.'
        elif lang=='fr': text=f'Pour {anchor}, la valeur actuelle est {q1} ; la nouvelle cible de production est {q2}.'
        else: text=f'Pour {anchor}, current value : {q1}; new production target : {q2}.'
    else:
        if lang=='en': text=f'For {anchor}, an earlier estimate was {q1}; the planning discussion also mentions {q2} for the same requirement.'
        elif lang=='fr': text=f'Pour {anchor}, une estimation précédente était {q1} ; la discussion de planification mentionne aussi {q2} pour la même exigence.'
        else: text=f'Pour {anchor}, earlier estimate : {q1}; planning mentionne aussi {q2} pour le même requirement.'
    return text,roles

_CONTEXT_PARTS={
'en':{
 'phase':['capacity review','architecture workshop','production sizing','design validation','storage planning','requirements review','deployment preparation','performance planning'],
 'workload':['AI training','CFD','genomics','checkpointing','analytics','simulation','research data','mixed HPC','MPI production','large-scale post-processing'],
 'env':['production cluster','shared HPC platform','Lustre service','compute environment','research cluster','storage platform'],
 'team':['storage team','HPC architects','operations team','application owners','platform engineers','capacity planners'],
 'stage':['initial estimate','technical review','final sizing pass','migration planning','procurement preparation','acceptance planning','deployment design','capacity refresh'],
 'site':['primary datacenter','research facility','shared service','production environment','compute center','HPC program']},
'fr':{
 'phase':['revue de capacité','atelier d’architecture','dimensionnement de production','validation du design','planification stockage','revue des besoins','préparation du déploiement','planification des performances'],
 'workload':['entraînement IA','CFD','génomique','checkpointing','analytique','simulation','données de recherche','charge HPC mixte','production MPI','post-traitement massif'],
 'env':['cluster de production','plateforme HPC partagée','service Lustre','environnement de calcul','cluster de recherche','plateforme de stockage'],
 'team':['équipe stockage','architectes HPC','équipe exploitation','responsables applicatifs','ingénieurs plateforme','équipe capacité'],
 'stage':['estimation initiale','revue technique','dimensionnement final','planification de migration','préparation des achats','préparation de recette','conception du déploiement','mise à jour de capacité'],
 'site':['datacenter principal','centre de recherche','service partagé','environnement de production','centre de calcul','programme HPC']},
'mixed':{
 'phase':['capacity review','atelier architecture','production sizing','design validation','storage planning','requirements review'],
 'workload':['AI training','CFD workload','génomique','checkpointing HPC','analytics workload','simulation MPI','research data'],
 'env':['production cluster','plateforme HPC','Lustre service','compute environment','storage platform'],
 'team':['storage team','architectes HPC','ops team','application owners','platform engineers'],
 'stage':['initial sizing','revue technique','final sizing','migration planning','procurement prep','deployment design'],
 'site':['primary datacenter','research facility','shared service','production environment','compute center']}
}

def unique_context(split,lang,ordinal):
    d=_CONTEXT_PARTS[lang]
    lens=[len(d['phase']),len(d['workload']),len(d['env']),len(d['team']),len(d['stage']),len(d['site'])]
    x=ordinal
    vals=[]
    for n in lens:
        vals.append(x % n); x//=n
    phase=d['phase'][vals[0]]; workload=d['workload'][vals[1]]; env=d['env'][vals[2]]; team=d['team'][vals[3]]; stage=d['stage'][vals[4]]; site=d['site'][vals[5]]
    if lang=='en':
        lead={'train':'During','val':'As part of','test':'While finalizing'}[split]
        return f'{lead} the {phase} for the {workload} workload on the {env} at the {site}; team: {team}; stage: {stage}. Requirement: '
    if lang=='fr':
        lead={'train':'Contexte','val':'Contexte de validation','test':'Contexte de test final'}[split]
        return f'{lead} — {phase}; charge : {workload}; environnement : {env}; site : {site}; équipe : {team}; étape : {stage}. Besoin : '
    lead={'train':'Mixed context','val':'Mixed validation context','test':'Mixed final-test context'}[split]
    return f'{lead} — phase: {phase}; workload: {workload}; environment: {env}; site: {site}; team: {team}; stage: {stage}. Requirement: '

def add_unique_context(text,split,lang,ordinal,category):
    # Keep short/telegraphic samples short but still unique with a compact realistic qualifier.
    if category=='short_telegraphic':
        d=_CONTEXT_PARTS[lang]
        x=ordinal
        vals=[]
        for key in ('workload','env','stage','team','site'):
            vals.append(x % len(d[key])); x//=len(d[key])
        workload=d['workload'][vals[0]]; env=d['env'][vals[1]]; stage=d['stage'][vals[2]]; team=d['team'][vals[3]]; site=d['site'][vals[4]]
        tag={'train':'prod note','val':'validation note','test':'final-design note'}[split]
        if lang=='en': return f'{text} ({tag}; {workload}; {env}; {stage}; {team}; {site}).'
        if lang=='fr': return f'{text} ({tag} ; {workload} ; {env} ; {stage} ; {team} ; {site}).'
        return f'{text} ({tag}; {workload}; {env}; {stage}; {team}; {site}).'
    return unique_context(split,lang,ordinal)+text

def find_standalone(text, raw):
    start=0
    while True:
        pos=text.find(raw,start)
        if pos<0: return -1
        before=text[pos-1] if pos>0 else ''
        after=text[pos+len(raw)] if pos+len(raw)<len(text) else ''
        # Numeric quantities must not be matched as substrings of a larger number.
        prevprev=text[pos-2] if pos>1 else ''
        nextnext=text[pos+len(raw)+1] if pos+len(raw)+1<len(text) else ''
        left_decimal = (before=='.' and prevprev.isdigit())
        right_decimal = (after=='.' and nextnext.isdigit())
        left_ok = not (raw and raw[0].isdigit() and (before.isdigit() or left_decimal))
        right_ok = not (raw and raw[-1].isdigit() and (after.isdigit() or right_decimal))
        if left_ok and right_ok: return pos
        start=pos+1

def make_pair_samples(field,category,split,group_index,lang,rng):
    if field=='__UNRESOLVED__':
        # use unitful network/power/capacity diagnostics to make hard negatives
        choices=[('throughput',f"{rng.choice([25,50,100])} GB/s",'GB/s'),('power',f"{rng.choice([8,12,15])} kW",'kW'),('capacity',f"{rng.choice([200,500,800])} TiB",'TiB')]
        d1,q1,u1=rng.choice(choices); d2,q2,u2=rng.choice(choices)
        while q2==q1: d2,q2,u2=rng.choice(choices)
        v1=float(re.findall(r'[0-9.]+',q1)[0]); v2=float(re.findall(r'[0-9.]+',q2)[0]); v1=int(v1) if v1.is_integer() else v1; v2=int(v2) if v2.is_integer() else v2
        if category=='alternative_pair': text=alt_text(field,lang,q1,q2,'unspecified',rng); roles=('unspecified','unspecified')
        elif category=='current_target_collision': text,roles=role_collision_pair(field,lang,q1,q2,rng)
        else: text,roles=role_temporal_pair(field,lang,q1,q2,rng)
        dims=(d1,d2); units=(u1,u2); vals=(v1,v2)
    else:
        v1,v2=rng.sample(VALUES[field],2)
        if category!='alternative_pair': v1,v2=sorted((v1,v2))
        q1,val1,u1=qspec(field,v1,rng)
        q2,val2,u2=qspec(field,v2,rng)
        role=role_for(field,rng)
        if category=='alternative_pair': text=alt_text(field,lang,q1,q2,role,rng); roles=(role,role)
        elif category=='current_target_collision': text,roles=role_collision_pair(field,lang,q1,q2,rng)
        else: text,roles=role_temporal_pair(field,lang,q1,q2,rng)
        dims=(DIM[field],DIM[field]); units=(u1,u2); vals=(val1,val2)
    # Mark q1 and q2 separately in the same original text.
    # Use first exact occurrence, quantities chosen distinct.
    outs=[]
    for qi,(raw,val,unit,dim,role) in enumerate([(q1,vals[0],units[0],dims[0],roles[0]),(q2,vals[1],units[1],dims[1],roles[1])],1):
        pos=find_standalone(text,raw)
        if pos<0: raise RuntimeError((text,raw))
        marked=text[:pos]+f'[Q]{raw}[/Q]'+text[pos+len(raw):]
        outs.append((text,marked,raw,val,unit,dim,role,False,qi))
    return outs

def lang_schedule(n,category):
    if category=='mixed_code_switch': return ['mixed']*n
    # exact 40/40/20 where possible
    en=int(n*0.4); fr=int(n*0.4); mix=n-en-fr
    arr=['en']*en+['fr']*fr+['mixed']*mix
    return arr

def make_split(split):
    rng=random.Random(SEED+{'train':1,'val':2,'test':3}[split])
    rows=[]; sample_no=0
    for fidx,field in enumerate(FIELDS):
        for cidx,cat in enumerate(CATEGORIES):
            n=COUNT[split]
            langs=lang_schedule(n,cat); rng.shuffle(langs)
            if cat in ('alternative_pair','complex_role_temporal','current_target_collision'):
                # generate pair rows. n must be even.
                assert n%2==0
                for g in range(n//2):
                    lang=langs[2*g]
                    actual_cat=cat
                    pair=make_pair_samples(field,actual_cat,split,g,lang,rng)
                    group=f'grp_v3_{split}_{fidx:02d}_{cidx:02d}_{g:05d}'
                    for item in pair:
                        original,marked,raw,val,unit,dim,role,corrected,qi=item
                        ctx=unique_context(split,lang,(fidx*100000)+(cidx*10000)+g)
                        original=ctx+original; marked=ctx+marked
                        sample_no+=1
                        rows.append({
                            'sample_id':f'sem_v3_{split}_{sample_no:07d}', 'group_id':group,
                            'original_text':original,'marked_text':marked,'quantity_id':f'q{qi}','raw_quantity':raw,
                            'normalized_quantity':normalized_quantity(val,unit),'value':val,'unit':unit,'dimension':dim,'corrected':corrected,
                            'previous_question':None,'field':field,'role':role,'language':lang,
                            'source':f'synthetic_global_stress_v3+{cat}'})
            else:
                for i in range(n):
                    lang=langs[i]
                    original,marked,raw,val,unit,dim,role,corrected=make_regular(field,cat,split,i,lang,rng)
                    original=add_unique_context(original,split,lang,(fidx*100000)+(cidx*10000)+i,cat)
                    marked=add_unique_context(marked,split,lang,(fidx*100000)+(cidx*10000)+i,cat)
                    sample_no+=1
                    rows.append({
                        'sample_id':f'sem_v3_{split}_{sample_no:07d}', 'group_id':f'grp_v3_{split}_{fidx:02d}_{cidx:02d}_{i:05d}',
                        'original_text':original,'marked_text':marked,'quantity_id':'q1','raw_quantity':raw,
                        'normalized_quantity':normalized_quantity(val,unit),'value':val,'unit':unit,'dimension':dim,'corrected':corrected,
                        'previous_question':None,'field':field,'role':role,'language':lang,
                        'source':f'synthetic_global_stress_v3+{cat}'})
    rng.shuffle(rows)
    return rows

def load_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def write_jsonl(p,rows):
    with p.open('w',encoding='utf8',newline='\n') as fh:
        for r in rows: fh.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

stress={s:make_split(s) for s in ('train','val','test')}
# Preserve the original V1 curated TRAIN only; do not contaminate new val/test with previous splits.
v1=load_jsonl(V1_TRAIN)
# Prefix IDs/groups to guarantee uniqueness without changing content.
for i,r in enumerate(v1,1):
    r=dict(r); r['sample_id']=f'v1curated_{i:05d}_{r.get("sample_id","sample")}'
    r['group_id']=f'v1curated_train_{i:05d}_{r.get("group_id","group")}'
    r['source']=str(r.get('source','synthetic_curated_v1'))+'+preserved_in_v3_train'
    v1[i-1]=r
train_combined=v1+stress['train']
random.Random(SEED+77).shuffle(train_combined)

paths={
 'train':ROOT/'semantic_linker_train_v3_combined.jsonl',
 'train_stress':ROOT/'semantic_linker_train_v3_stress_only.jsonl',
 'val':ROOT/'semantic_linker_val_v3_global_stress.jsonl',
 'test':ROOT/'semantic_linker_test_v3_global_stress.jsonl'}
write_jsonl(paths['train'],train_combined); write_jsonl(paths['train_stress'],stress['train']); write_jsonl(paths['val'],stress['val']); write_jsonl(paths['test'],stress['test'])

# QA audits
expected_dim=DIM
allowed_roles=ROLES
qa={'status':'PASS','checks':{},'warnings':[]}
allsets={'train':train_combined,'val':stress['val'],'test':stress['test']}

def check(cond,name,detail=None):
    qa['checks'][name]={'pass':bool(cond),'detail':detail}
    if not cond: qa['status']='FAIL'

for split,rows in allsets.items():
    check(len(rows)==len({r['sample_id'] for r in rows}),f'{split}_unique_sample_ids')
    check(all(r['marked_text'].count('[Q]')==1 and r['marked_text'].count('[/Q]')==1 for r in rows),f'{split}_one_quantity_marker')
    check(all(f"[Q]{r['raw_quantity']}[/Q]" in r['marked_text'] for r in rows),f'{split}_raw_quantity_matches_marker')
    check(all(r['field'] in FIELDS for r in rows),f'{split}_field_contract')
    check(all(r['role'] in allowed_roles[r['field']] for r in rows),f'{split}_role_compatibility')
    check(all((r['field']=='__UNRESOLVED__' or r['dimension']==expected_dim[r['field']]) for r in rows),f'{split}_dimension_compatibility')
    check(all(r['language'] in ('en','fr','mixed') for r in rows),f'{split}_language_contract')
    check(len(rows)==len({r['marked_text'] for r in rows}),f'{split}_unique_marked_texts',{'n':len(rows),'unique':len({r['marked_text'] for r in rows})})
    # Quantity.value must be numeric in the source unit, matching the normalized quantity prefix.
    check(all((not r['source'].startswith('synthetic_global_stress_v3')) or str(int(r['value']) if isinstance(r['value'],float) and r['value'].is_integer() else r['value']) in r['normalized_quantity'] for r in rows),f'{split}_value_source_unit_contract')

# semantic sanity checks for role-sensitive text
for split,rows in allsets.items():
    role_bad=[]
    for r in rows:
        t=r['original_text'].lower()
        if r['field'] in ('target_read_gbps','target_write_gbps') and r['role']=='current_value' and 'target read throughput' in t:
            role_bad.append(r['sample_id'])
        if r['field']=='max_power_w' and r['role']=='current_value' and 'storage power limit' in t:
            role_bad.append(r['sample_id'])
    check(len(role_bad)==0,f'{split}_role_wording_sanity',role_bad[:20])

# lightweight grammar/sanity audit for known generator failure patterns
banned_patterns=[
    r'\bDuring la\b', r'\bthe architectes\b', r"\bl’architectes\b", r'\bLe capacité\b', r'\bLa stockage\b',
    r'\bLe bande passante\b', r'\bLa ratio\b', r'\bLe population\b', r'\bLa plus gros\b', r'\bune taux\b',
    r'\bLe puissance\b', r'\bLa plafond\b', r'\bLe limite\b', r'\bI think please\b', r'\bI think que minimum\b', r'\bTiB CPU cores\b', r'\bkW CPU cores\b', r'\bUSD CPU cores\b', r'\bGB/s CPU cores\b'
]
for split,rows in allsets.items():
    bad=[]
    for r in rows:
        for pat in banned_patterns:
            if re.search(pat,r['original_text'],flags=re.IGNORECASE):
                bad.append({'sample_id':r['sample_id'],'pattern':pat,'text':r['original_text'][:240]}); break
    check(len(bad)==0,f'{split}_generator_grammar_sanity',bad[:20])

# overlaps
for a,b in [('train','val'),('train','test'),('val','test')]:
    A={r['marked_text'] for r in allsets[a]}; B={r['marked_text'] for r in allsets[b]}
    check(len(A&B)==0,f'no_marked_text_overlap_{a}_{b}',len(A&B))
    GA={r['group_id'] for r in allsets[a]}; GB={r['group_id'] for r in allsets[b]}
    check(len(GA&GB)==0,f'no_group_overlap_{a}_{b}',len(GA&GB))

# alt pair audit for stress splits
for split,rows in stress.items():
    groups=defaultdict(list)
    for r in rows:
        if r['source'].endswith('+alternative_pair'): groups[r['group_id']].append(r)
    ok=True; bad=[]
    for g,rs in groups.items():
        marker_ok=True
        for rr in rs:
            pos=find_standalone(rr['original_text'],rr['raw_quantity'])
            expected=(rr['original_text'][:pos]+f"[Q]{rr['raw_quantity']}[/Q]"+rr['original_text'][pos+len(rr['raw_quantity']):]) if pos>=0 else None
            marker_ok = marker_ok and (rr['marked_text']==expected if expected is not None else False)
        if len(rs)!=2 or rs[0]['original_text']!=rs[1]['original_text'] or rs[0]['field']!=rs[1]['field'] or rs[0]['raw_quantity']==rs[1]['raw_quantity'] or not marker_ok:
            ok=False; bad.append(g)
    check(ok,f'{split}_alternative_pair_two_extractable_quantities',{'groups':len(groups),'bad':bad[:10]})

# Known problematic phrases must NOT be exact test items; they may inspire training patterns, not contaminate test.
known=[
'Approximately 260 noeuds compute will mount Lustre.',
'There are 200 clents.',
'Il y aura 250 clints sur le système.',
'Write throughput could be 20 or 30 GB/s.',
'Maximum file size: 40 ou 60 GB.'
]
test_text={r['original_text'] for r in stress['test']}
check(all(k not in test_text for k in known),'known_holdout_failures_not_copied_into_new_test')

# category/field counts and summaries
meta={}
for split,rows in allsets.items():
    src_counts=Counter(r['source'].split('+')[-1] for r in rows)
    meta[split]={
        'n':len(rows),'fields':dict(Counter(r['field'] for r in rows)),
        'roles':dict(Counter(r['role'] for r in rows)),
        'languages':dict(Counter(r['language'] for r in rows)),
        'dimensions':dict(Counter(r['dimension'] for r in rows)),
        'stress_families':dict(src_counts),
        'unique_original_texts':len({r['original_text'] for r in rows}),
        'unique_marked_texts':len({r['marked_text'] for r in rows})}
# hashes
meta['files']={k:{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for k,p in paths.items()}
meta['generator_seed']=SEED
meta['design_note']='Alternative X-or-Y requirements are represented as TWO positive samples sharing one original_text/group_id, one [Q] target at a time; downstream guards decide which candidate survives.'
(ROOT/'semantic_linker_v3_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'semantic_linker_v3_QA_REPORT.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')

# Sample alternative-pair evidence CSV-like JSONL for quick inspection
alts=[]
for r in stress['train']:
    if r['source'].endswith('+alternative_pair'):
        alts.append(r)
        if len(alts)>=40: break
write_jsonl(ROOT/'alternative_pair_examples.jsonl',alts)

readme=f'''# Semantic Linker V3 — Global Stress Dataset\n\n## Purpose\nThis dataset is a new, large stress-oriented fine-tuning corpus for the Requirement Extractor V2 Semantic Linker. It covers **all 11 quantitative fields plus `__UNRESOLVED__`**, not only `client_count`.\n\n## Important design decision: X or Y\nAlternative expressions such as `20 or 30 GB/s` are **not labelled ambiguous at the Semantic Linker stage**. The Semantic Linker must associate **both quantities** with the correct field. Each original alternative sentence therefore appears as a pair of training/evaluation samples sharing the same `group_id`, with `[Q]...[/Q]` placed on one candidate at a time. A downstream deterministic guard is responsible for deciding whether one candidate, both, or neither should survive.\n\n## Sizes\n- Training stress-only: **{len(stress['train']):,}** rows\n- Training combined (V1 curated train + V3 stress): **{len(train_combined):,}** rows\n- Validation: **{len(stress['val']):,}** rows\n- Untouched V3 test: **{len(stress['test']):,}** rows\n\n## Stress families\n{chr(10).join('- '+c for c in CATEGORIES)}\n\n## Critical HPC distinctions included\nThe corpus includes collisions such as compute-node count vs Lustre-client count, network bandwidth vs Lustre throughput, archive/local capacity vs requested Lustre capacity, CPU/GPU/MPI/OST/OSS counts vs filesystem counts, current values vs targets, and several quantities in one message.\n\n## Language\nEnglish, French, and mixed EN/FR code-switching are included. Mixed-language examples are intentionally natural and domain-specific rather than random word replacement.\n\n## Typo generalization\nSemantic-anchor typos use different mutation operators in train, validation, and test to reduce memorization of exact misspellings.\n\n## Reproducibility / QA\nSee `semantic_linker_v3_QA_REPORT.json` and `semantic_linker_v3_metadata.json`. The QA checks cover schema consistency, quantity markers, field/role/dimension contracts, ID uniqueness, split leakage, group leakage, and the two-candidate invariant for alternative expressions.\n\n## Scientific limitation\nThis is a large **synthetic + curated** corpus. No finite dataset can guarantee coverage of every possible user utterance or be proven to have zero unknown defects. The included QA report verifies the known structural and semantic invariants used by this project. For publication-grade robustness, retaining a future small human-written external test set is still recommended.\n\n## Recommended training file\n`semantic_linker_train_v3_combined.jsonl`\n\nUse validation only for model selection/calibration. Use the V3 test once after freezing the model and thresholds.\n'''
(ROOT/'README.md').write_text(readme,encoding='utf8')

# Copy generator into package for reproducibility
shutil.copy2('/mnt/data/build_semantic_v3.py',ROOT/'generate_semantic_linker_v3.py')

# Zip all
zip_path=Path('/mnt/data/semantic_linker_v3_global_stress_dataset.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(ROOT.iterdir()): z.write(p,p.name)

print('TRAIN_COMBINED',len(train_combined))
print('TRAIN_STRESS',len(stress['train']))
print('VAL',len(stress['val']))
print('TEST',len(stress['test']))
print('QA',qa['status'])
print('ZIP',zip_path,zip_path.stat().st_size)
print('FIELDS train',Counter(r['field'] for r in stress['train']))
print('LANG train stress',Counter(r['language'] for r in stress['train']))
print('ALT groups train',sum(1 for k,v in defaultdict(list, { }).items()))
