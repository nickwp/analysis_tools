import json
from importlib.resources import files
import numpy as np


class PMTMapping:
    def __init__(self):
        # Load the JSON from the package's data folder
        json_path = files('analysis_tools.data').joinpath('PMT_Mapping.json')
        print("json path", json_path)
        with json_path.open('r') as file:
            self.pmt_data = json.load(file)["mapping"]

        cards = np.array([int(k) // 100 for k in self.pmt_data.keys()])
        channels = np.array([int(k) % 100 for k in self.pmt_data.keys()])

        slots = np.array([v // 100 for v in self.pmt_data.values()])
        positions = np.array([v % 100 for v in self.pmt_data.values()])

        unique_cards, _idx = np.unique(cards, return_index=True)
        unique_slots = slots[_idx]

        self.slot_from_card = np.full(133, -1)
        self.card_from_slot = np.full(133, -1)
        self.slot_from_card[unique_cards] = unique_slots
        self.card_from_slot[unique_slots] = unique_cards

        self.position_from_card_channel = np.full([133, 20], -1)
        self.channel_from_slot_position = np.full([133, 20], -1)
        self.position_from_card_channel[cards, channels] = positions
        self.channel_from_slot_position[slots, positions] = channels

    def get_slot_pmt_pos_from_card_pmt_chan(self, card_id, pmt_channel):
        slot_id = self.slot_from_card[card_id]
        pmt_pos = self.position_from_card_channel[card_id, pmt_channel]
        return slot_id, pmt_pos

    def get_card_pmt_chan_from_slot_pmt_pos(self, slot_id, pmt_pos):
        card_id = self.card_from_slot[slot_id]
        channel_id = self.channel_from_slot_position[slot_id, pmt_pos]
        return card_id, channel_id

    def get_card_from_slot(self, slot_id):
        return self.card_from_slot[slot_id]

    def get_slot_from_card(self, card_id):
        return self.slot_from_card[card_id]