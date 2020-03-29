import music21
from DatasetManager.config import get_config
from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, OrchestraIteratorGenerator


def list_instruments_name(score_iterator):
    parts = set()
    instruments = set()
    for arrangement_pair in score_iterator():
        for score in arrangement_pair:
            part_names, instru_names = get_names_score(score)
            parts = parts.union(part_names)
            instruments = instruments.union(instru_names)
    return parts, instruments


def list_instruments_name_pure_orch(score_iterator):
    parts = set()
    instruments = set()
    for score in score_iterator():
        part_names, instru_names = get_names_score(score['Orchestra'])
        parts = parts.union(part_names)
        instruments = instruments.union(instru_names)
    return parts, instruments


def get_names_score(score):
    list_instrus = []
    list_parts = []
    for part in score.parts:
        instru = part.getInstrument().instrumentName
        part_name = part.partName
        if (part_name is None) or (instru is None):
            continue
        list_instrus.append(instru)
        list_parts.append(part_name)
    return set(list_parts), set(list_instrus)


def write_sets(ss, out_file):
    ll = list(ss)
    ll.sort()
    with open(out_file, 'w') as ff:
        for elem in ll:
            ff.write(f'{elem}\n')
    return


if __name__ == '__main__':
    config = get_config()
    # database_path = f'{config["database_path"]}/Orchestration/arrangement_mxml'
    # subsets = [
    #     'imslp'
    # ]
    # score_iterator = ArrangementIteratorGenerator(
    #     arrangement_path=database_path,
    #     subsets=subsets
    # )
    # parts, instruments = list_instruments_name(
    #     score_iterator=score_iterator
    # )

    database_path = f'{config["database_path"]}/Orchestration/BACKUP/Kunstderfuge'
    score_iterator = OrchestraIteratorGenerator(f'{database_path}/Selected_works_clean_mxml', process_file=True)
    parts, instruments = list_instruments_name_pure_orch(score_iterator)

    dump_folder = config["dump_folder"]
    out_file = f'{dump_folder}/arrangement/parts.txt'
    write_sets(parts, out_file)
    out_file = f'{dump_folder}/arrangement/instruments.txt'
    write_sets(instruments, out_file)
