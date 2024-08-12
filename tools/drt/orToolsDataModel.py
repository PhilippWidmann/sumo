# -*- coding: utf-8 -*-
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.dev/sumo
# Copyright (C) 2021-2024 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    orToolsDataModel.py
# @author  Johannes Rummel
# @date    2024-03-13

"""
Data model for drtOrtools.py to solve a drt problem with the ortools routing solver.
"""
# needed for type alias in python < 3.9
from __future__ import annotations
import os
import sys
import typing
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math

# SUMO modules
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci  # noqa
import traci._person
import traci._simulation

SPEED_DEFAULT = 20  # default vehicle speed in m/s
VERBOSE_PHILIPP = False  # Todo Philipp: remove this and all references


class CostType(Enum):
    DISTANCE = 1
    TIME = 2


@dataclass
class Vehicle:
    """
    Represents a vehicle/route for the routing problem.
    """
    id_vehicle: str  # vehicle id/name from SUMO
    vehicle_index: int
    start_node: int = None
    end_node: int = None

    def get_person_capacity(self) -> int:
        return traci.vehicle.getPersonCapacity(self.id_vehicle)

    def get_type_ID(self) -> str:
        return traci.vehicle.getTypeID(self.id_vehicle)

    def get_edge(self) -> str:
        return traci.vehicle.getRoadID(self.id_vehicle)

    def get_person_id_list(self) -> list[str]:
        return traci.vehicle.getPersonIDList(self.id_vehicle)

    def get_energy_capacity(self, include_charging: bool) -> int:
        if include_charging:
            return int(float(traci.vehicle.getParameter(self.id_vehicle, 'device.battery.maximumBatteryCapacity')))
        else:
            return 0

    def get_current_energy(self, include_charging: bool) -> int:
        if include_charging:
            return int(float(traci.vehicle.getParameter(self.id_vehicle, 'device.battery.actualBatteryCapacity')))
        else:
            return 0


@dataclass
class Reservation:
    """
    Represents a request for a transportation.
    """
    reservation: traci._person.Reservation
    from_node: int = None
    to_node: int = None
    direct_route_cost: int = None
    current_route_cost: int = None
    vehicle: Vehicle = None

    def is_new(self) -> bool:
        if self.reservation.state == 1 or self.reservation.state == 2:
            return True
        else:
            return False

    def is_picked_up(self) -> bool:
        return self.reservation.state == 8

    def is_from_node(self, node: int) -> bool:
        return (not self.is_picked_up() and self.from_node == node)

    def is_to_node(self, node: int) -> bool:
        return self.to_node == node

    def get_from_edge(self) -> str:
        return self.reservation.fromEdge

    def get_to_edge(self) -> str:
        return self.reservation.toEdge

    def get_id(self) -> str:
        return self.reservation.id

    def get_persons(self) -> list[str]:
        return self.reservation.persons

    def update_direct_route_cost(self, type_vehicle: str, cost_matrix: list[list[int]] = None,
                                 cost_type: CostType = CostType.DISTANCE):
        if self.direct_route_cost:
            return
        if not self.is_picked_up():
            self.direct_route_cost = cost_matrix[self.from_node][self.to_node]
        else:
            # TODO: use 'historical data' from dict in get_cost_matrix instead
            route: traci._simulation.Stage = traci.simulation.findRoute(
                self.get_from_edge(), self.get_to_edge(), vType=type_vehicle)
            if cost_type == CostType.TIME:
                self.direct_route_cost = round(route.travelTime)
            elif cost_type == CostType.DISTANCE:
                self.direct_route_cost = round(route.length)
            else:
                raise ValueError(f"Cannot set given cost ({cost_type}).")

    def update_current_route_cost(self, cost_type: CostType = CostType.DISTANCE):
        person_id = self.reservation.persons[0]
        stage: traci._simulation.Stage = traci.person.getStage(person_id, 0)
        # stage type '3' is defined as 'driving'
        assert stage.type == 3
        if cost_type == CostType.DISTANCE:
            self.current_route_cost = round(stage.length)
        elif cost_type == CostType.TIME:
            self.current_route_cost = round(stage.travelTime)
        else:
            raise ValueError(f"Cannot set given cost ({cost_type}).")


@dataclass
class ChargingOpportunity:
    id_charging_station: str
    available_energy: int
    charging_time: int
    node: int = None
    vehicle: Vehicle = None

    def get_edge(self) -> str:
        return traci.chargingstation.getLaneID(self.id_charging_station).rsplit('_', 1)[0]

    def get_stopping_point_id(self) -> tuple[str, int]:
        lane = traci.chargingstation.getLaneID(self.id_charging_station)
        if (self.id_charging_station in traci.parkingarea.getIDList()
                and lane == traci.parkingarea.getLaneID(self.id_charging_station)):
            flag = 65  # Stop at the parking area of same name at same lane (64 for parkingArea, 1 for parkingState)
        else:
            flag = 32  # Stop at charging station directly
        return self.id_charging_station, flag



@dataclass
class ORToolsDataModel:
    """
    Data model class used by constrains of the OR-tools lib.
    """
    # nodeID of the depot
    depot: int
    cost_matrix: list[list[int]]
    time_matrix: list[list[int]]
    energy_matrix: list[list[int]]
    pickups_deliveries: list[Reservation]
    dropoffs: list[Reservation]
    num_vehicles: int
    starts: list[int]
    ends: list[int]
    demands: list[int]
    available_energy: list[int]
    vehicle_capacities: list[int]
    energy_capacities: list[int]
    drf: float
    waiting_time: int
    waiting_time_penalty: int
    time_windows: list[(int, int)]
    fix_allocation: bool
    max_time: int
    initial_routes: dict[int: list[list[int]]]
    penalty: int
    reservations: list[Reservation]
    charging_opportunities: list[ChargingOpportunity]
    vehicles: list[Vehicle]
    cost_type: CostType
    include_charging: bool

    def __str__(self):
        return f'number of vehicles: {self.num_vehicles}, ...'

    def get_penalty(self, explicitly_time_related: bool = False) -> int:
        """Returns penalty. If explicitly time related, it depends on the CostType of the data."""
        if not explicitly_time_related:
            return self.penalty
        if self.cost_type == CostType.DISTANCE:
            return round(self.penalty * SPEED_DEFAULT)
        else:
            return self.penalty


@dataclass
class Node:
    """
    Connects an object of the routing problem with a nodeID.
    """
    class NodeType(Enum):
        FROM_EDGE = 1
        TO_EDGE = 2
        VEHICLE = 3
        DEPOT = 4

    # id: int = field(default_factory=...)
    node_type: NodeType


# use 'type' statement in python version 3.12 or higher
NodeObject = typing.Union[str, Vehicle, Reservation, ChargingOpportunity]


def create_nodes(reservations: list[Reservation], vehicles: list[Vehicle],
                 charging_opportunities: list[ChargingOpportunity]) -> list[NodeObject]:
    """
    Sets the node ids from 0...n for the locations of the start and
    end points of the reservations and vehicles.
    """
    n = 0  # reserved for depot (which can also be a free location)
    node_objects = ['depot']
    n += 1
    for res in reservations:
        if not res.is_picked_up():
            node_objects.append(res)
            res.from_node = n
            n += 1
        else:
            res.from_node = None  # for clarity in debug
        node_objects.append(res)
        res.to_node = n
        n += 1
    for veh in vehicles:
        node_objects.append(veh)
        veh.start_node = n
        n += 1
        veh.end_node = 0  # currently all vehicles end at depot
        # TODO: to generalize the end nodes, separate nodes are needed
    for co in charging_opportunities:
        node_objects.append(co)
        co.node = n
        n += 1
    return node_objects


def create_vehicles(fleet: list[str]) -> list[Vehicle]:
    vehicles = []
    for i, veh_id in enumerate(fleet):
        veh = Vehicle(veh_id, i)
        vehicles.append(veh)
    return vehicles


def create_charging_opportunities(number_charging_duplicates: int, fleet: list[str]) -> list[ChargingOpportunity]:
    # Prepare allowed charging stops
    # Each charging stop needs its own node. Thus, we use multiple nodes per charging station.
    charging_opportunities = []
    if number_charging_duplicates > 0:
        charging_stations = traci.chargingstation.getIDList()
        energy_capacities = [int(float(traci.vehicle.getParameter(id_vehicle, 'device.battery.maximumBatteryCapacity')))
                             for id_vehicle in fleet]
        for cs in charging_stations:
            charging_duration = max(energy_capacities) / traci.chargingstation.getChargingPower(cs) * 60 * 60
            for i in range(number_charging_duplicates):
                charging_opp = ChargingOpportunity(
                    id_charging_station=cs,
                    available_energy=max(energy_capacities),
                    charging_time=charging_duration,
                )
                charging_opportunities.append(charging_opp)
    return charging_opportunities


# list[traci.Person.Reservation]
def create_new_reservations(data_reservations: list[Reservation]) -> list[Reservation]:
    """create Reservations that not already exist"""
    sumo_reservations = traci.person.getTaxiReservations(0)  # TODO: state 1 should be enough

    data_reservations_ids = [res.get_id() for res in data_reservations]
    new_reservations = []
    for res in sumo_reservations:
        if res.id not in data_reservations_ids:
            new_reservations.append(Reservation(res))
    return new_reservations


def update_reservations(data_reservations: list[Reservation]) -> list[Reservation]:
    """update the Reservation.reservation and also remove Reservations that are completed"""
    sumo_reservations: tuple[traci._person.Reservation] = traci.person.getTaxiReservations(0)
    updated_reservations = []
    for data_reservation in data_reservations:
        new_res = [res for res in sumo_reservations if res.id == data_reservation.get_id()]
        if new_res:
            data_reservation.reservation = new_res[0]
            updated_reservations.append(data_reservation)
    return updated_reservations


def reject_late_reservations(data_reservations: list[Reservation], waiting_time: int,
                             timestep: float) -> tuple[list[Reservation], list[Reservation]]:
    """
    rejects reservations that are not assigned to a vehicle and cannot be served by time

    Returns a cleared list and a list of the removed reservations.
    """
    new_data_reservations = []
    rejected_reservations = []
    for data_reservation in data_reservations:
        if not data_reservation.vehicle and data_reservation.reservation.reservationTime + waiting_time < timestep:
            #for person in data_reservation.get_persons():  # Todo Philipp: restore this and report bug; seem to only happen if passengers change at train/bus stops!
            #    traci.person.removeStages(person)
            rejected_reservations.append(data_reservation)
        else:
            new_data_reservations.append(data_reservation)
    return new_data_reservations, rejected_reservations


def map_vehicles_to_reservations(vehicles: list[Vehicle], reservations: list[Reservation]) -> None:
    """
    Sets the vehicle attribute of the reservations with the vehicle that contains the same persons.
    """
    for vehicle in vehicles:
        persons_in_vehicle = vehicle.get_person_id_list()
        for reservation in reservations:
            if reservation.get_persons()[0] in persons_in_vehicle:
                reservation.vehicle = vehicle


def get_edge_of_node_object(node_object: NodeObject, node: int) -> str | None:
    """
    Returns the edge of the given NodeObject. "node" is needed for Reservations,
    to make clear if the edge of the departure or destination is searched.
    Returns "None" if an edge cannot be found.
    """
    if isinstance(node_object, Vehicle) or isinstance(node_object, ChargingOpportunity):
        return node_object.get_edge()
    if isinstance(node_object, Reservation):
        if node_object.is_from_node(node):
            return node_object.get_from_edge()
        if node_object.is_to_node(node):
            return node_object.get_to_edge()
    return None


def get_demand_of_node_object(node_object: NodeObject, node: int) -> int | None:
    """
    Returns "None" if node is not from_node or to_node of a reservation.
    """
    if isinstance(node_object, str) and node_object == 'depot':
        return 0
    if isinstance(node_object, Vehicle):
        return traci.vehicle.getPersonNumber(node_object.id_vehicle)
    if isinstance(node_object, Reservation):
        if node_object.is_from_node(node):
            return 1
        if node_object.is_to_node(node):
            return -1
    if isinstance(node_object, ChargingOpportunity):
        return 0
    return None


def get_available_energy_of_node_object(node_object: NodeObject, include_energy: bool) -> int:
    if (isinstance(node_object, str) and node_object == 'depot'
            or isinstance(node_object, Reservation)):
        return 0
    if isinstance(node_object, Vehicle):
        return node_object.get_current_energy(include_energy)
    if isinstance(node_object, ChargingOpportunity):
        return node_object.available_energy


# TODO: If cost_type is TIME, remove cost_matrix and cost_dict.
def get_cost_matrix(node_objects: list[NodeObject], cost_type: CostType, include_charging: bool):
    """Get cost matrix between edges.
    Index in cost matrix is the same as the node index of the constraint solver."""

    # get vehicle type of one vehicle (we suppose all vehicles are of the same type)
    type_vehicle, id_vehicle = next(((x.get_type_ID(), x.id_vehicle)
                                    for x in node_objects if isinstance(x, Vehicle)), None)
    boardingDuration_param = traci.vehicletype.getBoardingDuration(type_vehicle)
    boardingDuration = 0 if boardingDuration_param == '' else round(float(boardingDuration_param))
    # TODO: pickup and dropoff duration of first vehicle is used for all vehicles!!!
    pickUpDuration_param = traci.vehicle.getParameter(id_vehicle, 'device.taxi.pickUpDuration')
    pickUpDuration = 0 if pickUpDuration_param == '' else round(float(pickUpDuration_param))
    dropOffDuration_param = traci.vehicle.getParameter(id_vehicle, 'device.taxi.dropOffDuration')
    dropOffDuration = 0 if dropOffDuration_param == '' else round(float(dropOffDuration_param))
    # estimate average energy consumption using all vehicles
    vehicles = [v for v in node_objects if isinstance(v, Vehicle)]
    energy_consumption_estimate = get_energy_consumption_estimate([v.id_vehicle for v in vehicles], include_charging)

    n_edges = len(node_objects)
    time_matrix = np.zeros([n_edges, n_edges], dtype=int)
    cost_matrix = np.zeros([n_edges, n_edges], dtype=int)
    energy_matrix = np.zeros([n_edges, n_edges], dtype=int)
    travel_time_dict = {}
    travel_distance_dict = {}
    # TODO initialize travel_distance_dict and travel_time_dict{} in run() and update for speed improvement
    for ii, from_node_object in enumerate(node_objects):
        edge_from = get_edge_of_node_object(from_node_object, ii)
        for jj, to_node_object in enumerate(node_objects):
            edge_to = get_edge_of_node_object(to_node_object, jj)
            # cost to depot should be always 0
            # (means there is no way to depot in the end)
            if ii == jj or from_node_object == 'depot' or to_node_object == 'depot':
                time_matrix[ii][jj] = 0
                cost_matrix[ii][jj] = 0
                energy_matrix[ii][jj] = 0
                if to_node_object == 'depot':
                    # Require high final charging level to avoid vehicles stranding somewhere
                    if isinstance(from_node_object, ChargingOpportunity):
                        energy_matrix[ii][jj] = - from_node_object.available_energy
                    else:
                        energy_matrix[ii][jj] = 1 * vehicles[0].get_energy_capacity(include_charging)
                continue

            # Compute and save travel time/distance if the combination of edges is new
            if (edge_from, edge_to) not in travel_time_dict:
                if edge_from == edge_to:
                    # This assumes that there is only one stopping point per edge and thus no travel is necessary
                    # as passengers are picked up/dropped off from the same place.
                    # Todo Philipp: This assumption makes sense for trains, but check for buses etc.
                    travel_time_dict[(edge_from, edge_to)] = 0
                    travel_distance_dict[(edge_from, edge_to)] = 0
                else:
                    route: traci._simulation.Stage = traci.simulation.findRoute(edge_from, edge_to,
                                                                                vType=type_vehicle)
                    travel_time_dict[(edge_from, edge_to)] = round(route.travelTime)
                    travel_distance_dict[(edge_from, edge_to)] = round(route.length)

            # Initialize matrices with baseline travel time
            time_matrix[ii][jj] = travel_time_dict[(edge_from, edge_to)]
            energy_matrix[ii][jj] = round(travel_distance_dict[(edge_from, edge_to)] * energy_consumption_estimate)

            # Add time at stop jj (depends on node, not just edge; so do not save this in travel_time_dict)
            # All stopping times are added to the travel time of the preceding edge
            if isinstance(to_node_object, Reservation):
                # Note: SUMO combines pickup and drop-off time of multiple reservations at the same stop,
                # but boardingTime is always added per entering/exiting passenger
                if edge_from == edge_to:
                    # fix stopping time was already added at the previous node
                    fix_stopping_time_jj = 0
                elif to_node_object.is_to_node(jj):
                    fix_stopping_time_jj = dropOffDuration
                elif to_node_object.is_from_node(jj):
                    fix_stopping_time_jj = pickUpDuration
                else:
                    raise ValueError('Node object is a reservation, but neither the from nor to node. '
                                     'This should not happen.')
                time_matrix[ii][jj] += max(fix_stopping_time_jj, boardingDuration)
            if isinstance(from_node_object, ChargingOpportunity):
                time_matrix[ii][jj] += from_node_object.charging_time
                energy_matrix[ii][jj] -= from_node_object.available_energy

            if cost_type == CostType.TIME:
                cost_matrix[ii][jj] = time_matrix[ii][jj]
            elif cost_type == CostType.DISTANCE:
                cost_matrix[ii][jj] = travel_distance_dict[(edge_from, edge_to)]
    return cost_matrix.tolist(), time_matrix.tolist(), energy_matrix.tolist()


def get_time_window_of_node_object(node_object: NodeObject, node: int, end: int) -> tuple[int, int]:
    """returns a pair with earliest and latest service time"""
    current_time = round(traci.simulation.getTime())
    max_time = round(end)

    time_window = None
    if isinstance(node_object, str) and node_object == 'depot':
        time_window = (current_time, max_time)
    elif isinstance(node_object, Vehicle):
        # TODO: throws an exception if not set: traci.vehicle.getParameter(node_object.id_vehicle, 'device.taxi.end')
        device_taxi_end = max_time
        time_window_end = max_time if device_taxi_end == '' else round(float(device_taxi_end))
        time_window = (current_time, time_window_end)
    elif isinstance(node_object, Reservation):
        person_id = node_object.get_persons()[0]
        if node_object.is_from_node(node):
            pickup_earliest = traci.person.getParameter(person_id, "pickup_earliest")
            if pickup_earliest:
                pickup_earliest = round(float(pickup_earliest))
            else:
                pickup_earliest = current_time
            time_window = (pickup_earliest, max_time)
        if node_object.is_to_node(node):
            dropoff_latest = traci.person.getParameter(person_id, "dropoff_latest")
            if dropoff_latest:
                dropoff_latest = round(float(dropoff_latest))
            else:
                dropoff_latest = max_time
            time_window = (current_time, dropoff_latest)
    elif isinstance(node_object, ChargingOpportunity):
        time_window = (current_time, max_time)
    else:
        raise ValueError(f"Cannot set time window for node {node}.")
    return time_window


def get_vehicle_by_vehicle_index(vehicles: list[Vehicle], index: int) -> Vehicle:
    for vehicle in vehicles:
        if vehicle.vehicle_index == index:
            return vehicle
    return None


def get_reservation_by_node(reservations: list[Reservation], node: int) -> Reservation:
    for reservation in reservations:
        if reservation.is_from_node(node) or reservation.is_to_node(node):
            return reservation
    return None


def get_penalty(penalty_factor: str | int, cost_matrix: list[list[int]]) -> int:
    if penalty_factor == 'dynamic':
        max_cost = max(max(sublist) for sublist in cost_matrix)
        return round_up_to_next_power_of_10(max_cost)
    else:
        return penalty_factor


# Todo Philipp: This global var is a hacky workaround for keeping previous values
#     because the distance counter returns error values for parked vehicles
PREVIOUS_DISTANCES = None
def get_energy_consumption_estimate(fleet: list[str], include_charging: bool) -> float:
    if not include_charging:
        return 0
    global PREVIOUS_DISTANCES
    if PREVIOUS_DISTANCES is None:
        PREVIOUS_DISTANCES = [0] * len(fleet)

    distances = []
    energy_used = 0
    for i, id_vehicle in enumerate(fleet):
        distances += [max(PREVIOUS_DISTANCES[i], traci.vehicle.getDistance(id_vehicle))]
        energy_used += float(traci.vehicle.getParameter(id_vehicle, "device.battery.totalEnergyConsumed"))
    if sum(distances) < 10000:
        Wh_per_m = 0
    else:
        Wh_per_m = energy_used / sum(distances)
    PREVIOUS_DISTANCES = distances
    print(Wh_per_m)
    #return 1.5 # Todo Philipp: Remove this!
    return Wh_per_m


def round_up_to_next_power_of_10(n: int) -> int:
    if n < 0:
        raise ValueError(f"Input '{n}' must be a positive integer")
    if n == 0:
        return 1
    # Determine the number of digits of the input value
    num_digits = math.floor(math.log10(n)) + 1
    scale = 10 ** (num_digits - 1)
    leading_digit = n // scale
    # If the input value is not already a power of 10, increase the leading digit by 1
    if n % scale != 0:
        leading_digit += 1
    rounded_value = leading_digit * scale
    return rounded_value
