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

# @file    ortools_pdp.py
# @author  Philip Ritzer
# @author  Johannes Rummel
# @date    2021-12-16

"""
Capacitated vehicle routing problem with pickup and delivery.

Based on https://developers.google.com/optimization/routing/pickup_delivery#complete_programs
"""

# needed for type alias in python < 3.9
from __future__ import annotations

import sys
from typing import List, Dict, Tuple

import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import orToolsDataModel

Node = int
Route = List[Node]
ORToolsSolution = Dict[int, Tuple[Route, int]]


def get_solution(data: orToolsDataModel.ORToolsDataModel, manager: pywrapcp.RoutingIndexManager,
                 routing: pywrapcp.RoutingModel, solution: pywrapcp.Assignment, verbose: bool) -> ORToolsSolution:
    """Returns the solution as a dict with one entry for each
    vehicle (vehicle id: [0, ..., n_veh-1]) including the
    route (list of nodes) and costs."""
    if verbose:
        print(f'Objective: {solution.ObjectiveValue()}')
    time_dimension = routing.GetDimensionOrDie('Time')
    if data.include_charging:
        energy_dimension = routing.GetDimensionOrDie('Energy')
    solution_dict = {}
    total_cost = 0
    for vehicle_id in range(data.num_vehicles):
        route = []
        plan_output = ''
        predicted_energy = []
        index = routing.Start(vehicle_id)
        if verbose:
            plan_output = 'Route for vehicle %s:\n    ' % vehicle_id
        route_cost = 0
        route_load = 0
        energy_value = 0
        energy_slack = 0
        energy_transit = 0
        while not routing.IsEnd(index):
            current_node = manager.IndexToNode(index)
            route_load += data.demands[current_node]
            time_var = time_dimension.CumulVar(index)
            if data.include_charging:
                energy_value = solution.Value(energy_dimension.CumulVar(index))
                energy_slack = solution.Value(energy_dimension.SlackVar(index))
                next_index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(next_index)
                energy_transit = data.energy_matrix[current_node][next_node]
                predicted_energy.append(energy_value)
            if verbose or orToolsDataModel.VERBOSE_PHILIPP:
                plan_output += (' %s (L: %s, C: %s, T: (%s,%s), E: %s (-> ES: %s, ET: %s ))\n -> ' %
                                (current_node, route_load, route_cost, solution.Min(time_var), solution.Max(time_var),
                                 energy_value, energy_slack, energy_transit))
            route.append(current_node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        last_node = manager.IndexToNode(index)
        route_load += data.demands[last_node]
        time_var = time_dimension.CumulVar(index)
        if data.include_charging:
            energy_value = solution.Value(energy_dimension.CumulVar(index))
            predicted_energy.append(energy_value)
        route.append(last_node)
        if verbose or orToolsDataModel.VERBOSE_PHILIPP:
            plan_output += (' %s (L: %s, C: %s, T: (%s,%s), E: %s)\n' %
                            (last_node, route_load, route_cost, solution.Min(time_var), solution.Max(time_var),
                             energy_value))
            plan_output += 'Costs of the route: %s\n' % route_cost
            print(plan_output)
        total_cost += route_cost
        solution_dict[vehicle_id] = (route, route_cost)
    if verbose or orToolsDataModel.VERBOSE_PHILIPP:
        print(f'Total cost of the routes: {total_cost}')
        print(f'Objective value: {solution.ObjectiveValue()}')
    return solution_dict


TransitCallbackIndex = int


def set_travel_cost(data: orToolsDataModel.ORToolsDataModel, routing: pywrapcp.RoutingModel,
                    manager: pywrapcp.RoutingIndexManager, verbose: bool) -> TransitCallbackIndex:
    """Create and register a transit callback."""

    def distance_callback(from_index, to_index) -> int:
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data.cost_matrix[from_node][to_node]

    if verbose:
        print(' Register distance callback.')
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    return transit_callback_index


def add_cost_constraint(data: orToolsDataModel.ORToolsDataModel, routing: pywrapcp.RoutingModel,
                        transit_callback_index: TransitCallbackIndex, verbose: bool) -> pywrapcp.RoutingDimension:
    # Add costs/distance constraint.
    if verbose:
        print(' Add distance constraints...')
    matrix_costs = int(np.sum(data.cost_matrix))
    dimension_name = 'Costs'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        matrix_costs,  # TODO reasonable max costs/distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # the following tries to reduce the route costs of the vehicle with maximum costs
    # distance_dimension.SetGlobalSpanCostCoefficient(0)
    # the following tries to reduce the sum of all route costs
    # distance_dimension.SetSpanCostCoefficientForAllVehicles(0)
    return distance_dimension


def add_transportation_requests_constraint(data: orToolsDataModel.ORToolsDataModel,
                                           routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                                           solver: pywrapcp.Solver,
                                           distance_dimension: pywrapcp.RoutingDimension, verbose: bool):
    if verbose:
        print(' Add pickup and delivery constraints...')
    for request in data.pickups_deliveries:
        pickup_index = manager.NodeToIndex(request.from_node)
        delivery_index = manager.NodeToIndex(request.to_node)
        if verbose:
            print(f'pickup/dropoff nodes: {request.from_node}/{request.to_node}')
        routing.AddPickupAndDelivery(pickup_index, delivery_index)  # helps the solver
        # use same veh for pickup and dropoff
        solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        solver.Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))  # define order: first pickup then dropoff
        if request.is_new():  # is that a new request?
            # allows to reject the order but gives penalty
            if verbose:
                print(f'allow to reject new reservation {request.get_id()}')
            # penalty equivalent to a day of forbidden tardiness
            routing.AddDisjunction([pickup_index, delivery_index], 24*3600*data.get_penalty(True), 2)


def add_travel_time_constraint(data: orToolsDataModel.ORToolsDataModel,
                               routing: pywrapcp.RoutingModel,
                               manager: pywrapcp.RoutingIndexManager,
                               solver: pywrapcp.Solver,
                               time_dimension: pywrapcp.RoutingDimension, verbose: bool):
    for request in data.pickups_deliveries + data.dropoffs:
        earliest_pickup = request.reservation.depart
        direct_route_time = request.direct_route_time
        target_dropoff_time = int(earliest_pickup + data.waiting_time + data.drf * direct_route_time)
        dropoff_index = manager.NodeToIndex(request.to_node)
        time_dimension.SetCumulVarSoftUpperBound(
            dropoff_index,
            target_dropoff_time,
            data.waiting_time_penalty,  # cost: 1 sec tardiness per passenger = penalty meters driven by vehicle
        )


def add_direct_route_factor_constraint(data: orToolsDataModel.ORToolsDataModel,
                                       routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                                       solver: pywrapcp.Solver,
                                       distance_dimension: pywrapcp.RoutingDimension, verbose: bool):
    if data.drf == -1:
        return
    if verbose:
        print(' Add direct route factor constraints...')
    for request in data.pickups_deliveries:
        # hard constraint for new requests:
        if request.is_new():
            add_hard_direct_route_factor_constraint(request, data, manager, solver, distance_dimension, verbose)
        # soft constraint for old (know) requests:
        else:
            add_soft_direct_route_factor_constraint(routing, data, request, solver,
                                                    manager, distance_dimension, verbose)

    # if possible, let the route costs of the dropoffs less than the drf allows,
    # else minimize the route costs (in case the costs became larger than expected)
    for request in data.dropoffs:  # TODO: test needed
        direct_route_cost = request.direct_route_cost
        direct_route_cost_drf = solver.IntConst(round(direct_route_cost * data.drf))
        delivery_index = manager.NodeToIndex(request.to_node)
        distance_dimension.SetCumulVarSoftUpperBound(delivery_index, round(
            direct_route_cost * data.drf - request.current_route_cost), 100*data.get_penalty())
        if verbose:
            print(f"reservation {request.get_id()}: direct route cost {direct_route_cost} and "
                  f"(soft) max cost {direct_route_cost_drf.Value()}, already used costs {request.current_route_cost}")


def add_hard_direct_route_factor_constraint(request: orToolsDataModel.Reservation,
                                            data: orToolsDataModel.ORToolsDataModel,
                                            manager: pywrapcp.RoutingIndexManager,
                                            solver: pywrapcp.Solver,
                                            distance_dimension: pywrapcp.RoutingDimension, verbose: bool):
    pickup_index = manager.NodeToIndex(request.from_node)
    delivery_index = manager.NodeToIndex(request.to_node)
    # let the route cost be less or equal the direct route cost times drf
    direct_route_cost = request.direct_route_cost
    solver.Add(
        distance_dimension.CumulVar(delivery_index) - distance_dimension.CumulVar(pickup_index) <=  # route cost
        solver.IntConst(round(direct_route_cost * data.drf)))  # direct route cost times direct route factor
    direct_route_cost_drf = solver.IntConst(round(direct_route_cost * data.drf))
    if verbose:
        print(f"reservation {request.get_id()}: direct route cost {direct_route_cost} and "
              f"(hard) max cost {direct_route_cost_drf.Value()}")


def add_soft_direct_route_factor_constraint(routing: pywrapcp.RoutingModel,
                                            data: orToolsDataModel.ORToolsDataModel,
                                            request: orToolsDataModel.Reservation,
                                            solver: pywrapcp.Solver, manager: pywrapcp.RoutingIndexManager,
                                            distance_dimension: pywrapcp.RoutingDimension, verbose: bool):
    '''If the costs changed and the drf-constraint cannot be hold anymore use a soft constraint.
    '''
    matrix_costs = int(np.sum(data.cost_matrix))
    # Add new dimension only for this request:
    request_cost_dimension_name = f'request_cost_{request.get_id()}'
    routing.AddConstantDimensionWithSlack(
        0,             # Transition is 0
        matrix_costs,  # reasonable maximum request costs
        matrix_costs,  # reasonable maximum slack
        True,          # force start request costs with 0
        request_cost_dimension_name)
    request_cost_dimension: pywrapcp.RoutingDimension = routing.GetDimensionOrDie(request_cost_dimension_name)
    pickup_index = manager.NodeToIndex(request.from_node)
    delivery_index = manager.NodeToIndex(request.to_node)
    route_start = distance_dimension.CumulVar(pickup_index)
    route_end = distance_dimension.CumulVar(delivery_index)
    route_cost = request_cost_dimension.CumulVar(delivery_index)
    solver.Add(route_cost == route_end - route_start)
    request_cost_dimension.SetCumulVarSoftUpperBound(
        delivery_index,
        round(request.direct_route_cost * data.drf),
        100*data.get_penalty()
    )
    if verbose:
        print(f"reservation {request.get_id()}: direct route cost {request.direct_route_cost} and "
              f"(soft) max cost {int(request.direct_route_cost * data.drf)}")


def add_dropoff_constraint(data: orToolsDataModel.ORToolsDataModel,
                           routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                           verbose: bool):
    if verbose:
        print(' Add dropoff constraints...')
    # for veh_index, do_list in enumerate(data['dropoffs']):
    #    if verbose:
    #        print('vehicle %s with %s dropoffs' % (veh_index, len(do_list)))
    #    for do in do_list:
    #        index = manager.NodeToIndex(do[0])
    #        # start node
    #        veh_node = data['starts'][veh_index]
    #        if verbose:
    #            print('vehicle %s (%s), dropoff %s (%s), res_id %s' % (veh_index, veh_node, do[0], index, do[1]))
    #        #routing.VehicleVar(index).SetValues([-1,veh_index])
    #        routing.SetAllowedVehiclesForIndex([veh_index],index)
    for res in data.dropoffs:
        if verbose:
            print(f'reservation {res.get_id()} in veh {res.vehicle.id_vehicle}({res.vehicle.vehicle_index}), '
                  f'droppoff node: {res.to_node}')
        index = manager.NodeToIndex(res.to_node)
        routing.SetAllowedVehiclesForIndex([res.vehicle.vehicle_index], index)


def add_allocation_constraint(data: orToolsDataModel.ORToolsDataModel,
                              routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                              verbose: bool):
    if verbose:
        print(' Add "no re-allocation" constraints...')
    for res in data.pickups_deliveries:
        if res.vehicle:  # hasattr(res, 'vehicle_index'):
            if verbose:
                print(f'reservation {res.get_id()} in veh id={res.vehicle.vehicle_index}')
            index = manager.NodeToIndex(res.to_node)
            routing.SetAllowedVehiclesForIndex([res.vehicle.vehicle_index], index)


def add_capacity_constraint(data: orToolsDataModel.ORToolsDataModel,
                            routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                            verbose: bool):
    if verbose:
        print(' Add capacity constraints...')

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data.demands[from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    dimension_name = 'Capacity'
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data.vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        dimension_name)
    capacity_dimension = routing.GetDimensionOrDie(dimension_name)
    return capacity_dimension


def create_time_dimension(data: orToolsDataModel.ORToolsDataModel,
                          routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                          verbose: bool) -> pywrapcp.RoutingDimension:
    if verbose:
        print(' Create time dimension.')

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data.time_matrix[from_node][to_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    # routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
    dimension_name = 'Time'
    routing.AddDimension(
        time_callback_index,
        int(data.max_time),  # allow waiting time
        int(data.max_time),  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        dimension_name)
    time_dimension = routing.GetDimensionOrDie(dimension_name)
    return time_dimension


def add_time_windows_constraint(data: orToolsDataModel.ORToolsDataModel,
                                time_dimension: pywrapcp.RoutingDimension, manager: pywrapcp.RoutingIndexManager,
                                verbose: bool):
    if verbose:
        print(' Add time windows constraints...')

    depot = data.depot
    new_requests_nodes = [node for req in data.pickups_deliveries
                          for node in (req.from_node, req.to_node) if req.is_new()]
    old_requests_nodes = ([req.to_node for req in data.dropoffs] +
                          [node for req in data.pickups_deliveries
                           for node in (req.from_node, req.to_node) if not req.is_new()])
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data.time_windows):
        # no time window for depot:
        if location_idx == depot:
            continue
        index = manager.NodeToIndex(location_idx)
        # hard time window for vehicles and new requests:
        if location_idx in data.starts + new_requests_nodes:
            if verbose:
                print(f'hard time window for node {location_idx}: [{time_window[0]}, {time_window[1]}]')
            time_dimension.CumulVar(index).SetRange(
                time_window[0], time_window[1])  # TODO: check if set, else ignore it
        # soft time window for old requests:
        if location_idx in old_requests_nodes:
            if verbose:
                print(f'soft time window for node {location_idx}: [{time_window[0]}, {time_window[1]}]')
            time_dimension.SetCumulVarSoftLowerBound(index, time_window[0], 100*data.get_penalty(True))
            time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], 100*data.get_penalty(True))
            # time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # TODO: check if the followwing is needed
    # # Add time window constraints for each vehicle start node.
    # depot_idx = data['depot']
    # for vehicle_id in range(data['num_vehicles']):
    #     index = routing.Start(vehicle_id)
    #     time_dimension.CumulVar(index).SetRange(
    #         data['time_windows'][depot_idx][0],
    #         data['time_windows'][depot_idx][1])
    # for i in range(data['num_vehicles']):
    #     routing.AddVariableMinimizedByFinalizer(
    #         time_dimension.CumulVar(routing.Start(i)))
    #     routing.AddVariableMinimizedByFinalizer(
    #         time_dimension.CumulVar(routing.End(i)))


def add_waiting_time_constraints(data: orToolsDataModel.ORToolsDataModel,
                                 manager: pywrapcp.RoutingIndexManager,
                                 time_dimension: pywrapcp.RoutingDimension,
                                 verbose: bool):
    """Adds the constraints related to the maximum waiting times of the requests.
    """
    global_waiting_time = data.waiting_time
    # -1 means no waiting time is used
    if global_waiting_time == -1:
        return
    if verbose:
        print(' Add waiting time constraints...')
    # for now, only a global waiting time for the pick up is introduced
    # TODO: add special constraints for latests arrival and earliest depart
    for request in data.pickups_deliveries:
        pickup_index = manager.NodeToIndex(request.from_node)
        reservation_time = request.reservation.reservationTime
        requested_pickup_time = request.get_earliest_pickup() or reservation_time
        maximum_pickup_time = round(request.get_dropoff_latest() or (requested_pickup_time + global_waiting_time))
        if data.waiting_time_penalty == -1:
            # add hard constraint for new reservations
            if request.is_new():
                if verbose:
                    print(f"reservation {request.get_id()} has a maximum (hard) pickup time at {maximum_pickup_time}")
                min_time_window = time_dimension.CumulVar(pickup_index).Min()
                maximum_pickup_time = maximum_pickup_time if min_time_window < maximum_pickup_time else min_time_window
                time_dimension.CumulVar(pickup_index).SetMax(maximum_pickup_time)
            # add soft constraint (with high penalty) for old reservations
            else:
                time_dimension.SetCumulVarSoftUpperBound(
                    pickup_index,
                    maximum_pickup_time,
                    1*data.get_penalty(True))  # cost = coefficient * (cumulVar - maximum_pickup_time)
                if verbose:
                    print(f"reservation {request.get_id()} has a maximum (soft) pickup time at {maximum_pickup_time}")
            # add very small punishment for being over the target waiting time
        else:
            # Soft waiting time constraint: Set a small (!) penalty for going over the waiting time
            time_dimension.SetCumulVarSoftUpperBound(
                    pickup_index,
                    maximum_pickup_time,
                    data.waiting_time_penalty,  # cost: 1 sec tardiness per passenger = penalty meters driven by vehicle
            )


def create_energy_dimension(data: orToolsDataModel.ORToolsDataModel,
                            routing: pywrapcp.RoutingModel,
                            manager: pywrapcp.RoutingIndexManager,
                            verbose: bool):
    # Recharging follows example from
    # https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/samples/cvrp_reload.py
    if verbose:
        print(' Create energy dimension.')

    def energy_callback(from_index, to_index):
        """Returns the energy used for travelling between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data.energy_matrix[from_node][to_node]

    energy_callback_index = routing.RegisterTransitCallback(energy_callback)

    dimension_name = 'Energy'
    routing.AddDimensionWithVehicleCapacity(
        energy_callback_index,
        sys.maxsize,  # Slack: Will be overwritten based on type
        [sys.maxsize]*data.num_vehicles,  # Capacity: No hard, only soft u.b.: add_soft_minimal_energy_constraints
        False,  # Don't force start cumul to zero.
        dimension_name)
    energy_dimension = routing.GetDimensionOrDie(dimension_name)

    # Add Slack for resetting energy to max at charging stations.
    # Note that we have to fix the slack and not just give an upper bound to ensure correctness
    # e.g. vehicle with used energy 5/15 travels for 1 energy to charging station, then for 2 energy away from it
    # CumulVar(cs) = 5 + 1 + 0 = 6
    # CumulVar(after_cs) = 6 + (2-15)[travel-charge; in energy_matrix] + 9[slack] = 2
    # Other nodes will be set to 0 slack (energy consumption is fixed by travel).
    for co in data.charging_opportunities:
        index = manager.NodeToIndex(co.node)
        energy_dimension.SlackVar(index).SetMax(co.available_energy)
        # Todo Philipp: This only works as long as we charge to full
        routing.solver().Add(energy_dimension.SlackVar(index) >= co.available_energy - energy_dimension.CumulVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.CumulVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.SlackVar(index))
        # Visiting charging opportunities is optional
        # In particular we assume no passengers can enter/exit at charging opportunities
        # routing.AddDisjunction([index], 1)
        routing.AddToAssignment(energy_dimension.SlackVar(index))
    for vehicle in data.vehicles:
        # Start CumulVar with the used energy, no slack needed
        index = manager.NodeToIndex(vehicle.start_node)
        energy_dimension.CumulVar(index).SetValue(vehicle.get_energy_capacity(True) - vehicle.get_current_energy(True))
        energy_dimension.SlackVar(index).SetValue(0)
        routing.AddVariableMinimizedByFinalizer(energy_dimension.CumulVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.SlackVar(index))
        routing.AddToAssignment(energy_dimension.SlackVar(index))
    for res in data.pickups_deliveries:
        index = manager.NodeToIndex(res.from_node)
        energy_dimension.SlackVar(index).SetValue(0)
        routing.AddToAssignment(energy_dimension.SlackVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.CumulVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.SlackVar(index))

        index = manager.NodeToIndex(res.to_node)
        energy_dimension.SlackVar(index).SetValue(0)
        routing.AddVariableMinimizedByFinalizer(energy_dimension.CumulVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.SlackVar(index))
        routing.AddToAssignment(energy_dimension.SlackVar(index))
    for res in data.dropoffs:
        index = manager.NodeToIndex(res.to_node)
        energy_dimension.SlackVar(index).SetValue(0)
        routing.AddVariableMinimizedByFinalizer(energy_dimension.CumulVar(index))
        routing.AddVariableMinimizedByFinalizer(energy_dimension.SlackVar(index))
        routing.AddToAssignment(energy_dimension.SlackVar(index))
    return energy_dimension


def add_soft_minimal_energy_constraints(data: orToolsDataModel.ORToolsDataModel,
                                        energy_dimension: pywrapcp.RoutingDimension,
                                        routing: pywrapcp.RoutingModel,
                                        manager: pywrapcp.RoutingIndexManager):
    for node in range(len(data.available_energy)):
        if node == data.depot:
            continue
        index = manager.NodeToIndex(node)
        # Punish plans that would bring the vehicle below 20% remaining charge
        # Since simulation can differ from plan, vehicles may otherwise "strand"
        if node not in data.starts:
            energy_dimension.SetCumulVarSoftUpperBound(
                index,
                int(0.8*max(data.energy_capacities)),
                100000 * data.get_penalty(False)  # cost = coefficient * (cumulVar - critical_charge_level)
            )
            # energy_dimension.SetCumulVarSoftUpperBound(
            #     index,
            #     int(0.9 * max(data.energy_capacities)),
            #     100000  # cost = coefficient * (cumulVar - critical_charge_level)
            # )
        for index in [routing.End(v.vehicle_index) for v in data.vehicles]:
            # Vehicle must end at charging station to avoid stranding.
            # If it does, it will achieve energy level 0 at the depot
            energy_dimension.SetCumulVarSoftUpperBound(
                index,
                0,  # upper bound 0; this can always be reached if the last stop is a charging station
                100000 * data.get_penalty(False)  # cost = coefficient * (cumulVar - 0)
            )


def restrict_charging_to_unoccupied_vehicles_constraint(data: orToolsDataModel.ORToolsDataModel,
                                                        capacity_dimension: pywrapcp.RoutingDimension,
                                                        manager: pywrapcp.RoutingIndexManager):
    for co in data.charging_opportunities:
        index = manager.NodeToIndex(co.node)
        capacity_dimension.SetCumulVarSoftUpperBound(
            index,
            0,
            1000 * data.get_penalty(False),
        )


def solve_from_initial_solution(routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager,
                                search_parameters: any, data: orToolsDataModel.ORToolsDataModel, verbose: bool):
    solution_requests = data.initial_routes
    # get inital solution
    initial_routes = []
    if solution_requests is not None:
        for index_vehicle in solution_requests:
            # use request ids ([0]) here and align with current status of the requests
            request_order = solution_requests[index_vehicle][0].copy()
            unique_requests = []  # Todo: Make request_objects hashable to avoid for loop
            for request_object in solution_requests[index_vehicle][0]:
                if request_object not in unique_requests:
                    unique_requests.append(request_object)
            for request_object in unique_requests:
                if not isinstance(request_object, orToolsDataModel.Reservation):
                    continue
                # Check if some reservations have been completed and must be removed from the initial route
                # 0: done
                # 1: only drop-off left
                # 2: pick-up and drop-off left
                old_status = solution_requests[index_vehicle][0].count(request_object)
                if request_object in data.pickups_deliveries:
                    new_status = 2
                elif request_object in data.dropoffs:
                    new_status = 1
                else:
                    new_status = 0
                if new_status == 0:
                    # remove complete request
                    request_order = [req for req in request_order if req != request_object]
                if old_status == 2 and new_status == 1:
                    # remove first occurrence of the request
                    request_order.remove(request_object)
            # translate new requests order (ids) to nodes order
            # (e.g. [0,1,2,1,2] -> [0.to_node, 1.from_node, 2.from_node, 1.to_node, 2.to_node])
            # first occurrence from behind (will be "to_node")
            reversed_request_order = request_order.copy()
            reversed_request_order.reverse()  # e.g. [2,1,2,1,0]
            first_occurrence_from_behind = set([reversed_request_order.index(req)
                                                for req in request_order])  # e.g. [0,1,4]
            all_requests = data.pickups_deliveries.copy()
            all_requests.extend(data.dropoffs.copy())
            nodes_order = []
            for index, req in enumerate(reversed_request_order):
                if isinstance(req, orToolsDataModel.Reservation):
                    if index in first_occurrence_from_behind:
                        nodes_order.insert(0, manager.NodeToIndex(req.to_node))
                    else:
                        nodes_order.insert(0, manager.NodeToIndex(req.from_node))
                else:
                    nodes_order.insert(0, manager.NodeToIndex(req.node))
            # nodes_order = solution_requests[index_vehicle][2]  # [2] for nodes
            initial_routes.append(nodes_order)
    routing.CloseModelWithParameters(search_parameters)
    if verbose:
        print('Initial solution:')
        for index_vehicle, index_list in enumerate(initial_routes):
            print(f'veh {index_vehicle}: {[manager.IndexToNode(index) for index in index_list]}')
    initial_solution = routing.ReadAssignmentFromRoutes(initial_routes, True)
    solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
    return solution


def set_first_solution_heuristic(time_limit_seconds: int, verbose: bool) -> any:
    if verbose:
        print(' Set solution heuristic...')
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
    # search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC

    # GUIDED_LOCAL_SEARCH seems slow
    # search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC

    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    search_parameters.sat_parameters.num_search_workers = 8
    # search_parameters.lns_time_limit.seconds = 7
    # search_parameters.solution_limit = 100

    # Switch logging on for the search
    # search_parameters.log_search = True

    return search_parameters


def main(data: orToolsDataModel.ORToolsDataModel,
         time_limit_seconds: int = 10, verbose: bool = False, simplified_resolve: bool = False,
         ) -> ORToolsSolution | None:
    """Entry point of the program."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data.cost_matrix),  # number of locations (nodes)
        data.num_vehicles,  # number of vehicles
        data.starts, data.ends)  # start and end locations (nodes)

    # Create Routing Model.
    routing_parameters = pywrapcp.DefaultRoutingModelParameters()
    # routing_parameters.solver_parameters.trace_propagation = True
    # routing_parameters.solver_parameters.trace_search = True
    routing = pywrapcp.RoutingModel(manager, routing_parameters)

    # get solver
    solver = routing.solver()

    # define transit_callback and set travel cost
    transit_callback_index = set_travel_cost(data, routing, manager, verbose)
    time_dimension = create_time_dimension(data, routing, manager, verbose)
    # Add costs/distance constraint.
    distance_dimension = add_cost_constraint(data, routing, transit_callback_index, verbose)

    # Define Transportation Requests.
    add_transportation_requests_constraint(data, routing, manager, solver, distance_dimension, verbose)

    # Set direct route factor.
    add_direct_route_factor_constraint(data, routing, manager, solver, distance_dimension, verbose)

    # Force the vehicle to drop-off the reservations it already picked up
    add_dropoff_constraint(data, routing, manager, verbose)

    # If no reallocation is desired
    if data.fix_allocation:
        add_allocation_constraint(data, routing, manager, verbose)

    # Add Capacity constraint.
    capacity_dimension = add_capacity_constraint(data, routing, manager, verbose)

    # Add time window constraints.
    add_time_windows_constraint(data, time_dimension, manager, verbose)

    # Add waiting time constraints.
    add_waiting_time_constraints(data, manager, time_dimension, verbose)

    # Set travel time constraint
    add_travel_time_constraint(data, routing, manager, solver, time_dimension, verbose)

    # Setting first solution heuristic.
    search_parameters = set_first_solution_heuristic(time_limit_seconds, verbose)

    # Add all constraints belonging to energy tracking
    if data.include_charging:
        # Add dimension for used energy
        energy_dimension = create_energy_dimension(data, routing, manager, verbose)

        #for v in range(data['num_vehicles']):
        #    routing.SetVehicleUsedWhenEmpty(True, v)

        # Add penalty for planning trips with very low energy levels
        add_soft_minimal_energy_constraints(data, energy_dimension, routing, manager)

        # May only charge if there are no passengers in vehicle
        restrict_charging_to_unoccupied_vehicles_constraint(data, capacity_dimension, manager)

    # Solve the problem.
    if verbose:
        print('Start solving the problem.')
    if data.initial_routes:
        # TODO: change ids of the nodes, due to already picked up or droped off requests after last solution!
        solution = solve_from_initial_solution(routing, manager, search_parameters, data, verbose)
    else:
        solution = routing.SolveWithParameters(search_parameters)
#    solution = routing.SolveWithParameters(search_parameters)

    # Status codes: https://developers.google.com/optimization/routing/routing_options#search_status
    # Not listed is 7, which from some research should correspond to "global optimum found"
    print(f'## Optimization done, optimization status: {routing.status()}')
    if routing.status() not in [1, 2, 7]:  # [local opt, global opt]
        # If the heuristic cannot find a solution due to the added complexity of charging,
        # maybe it helps to ignore this and find a feasible solution without charging first.
        if data.include_charging and not simplified_resolve:
            print('+++ Retrying without energy constraints +++')
            data.include_charging = False
            # data.initial_routes = None
            solution_without_energy = main(data, time_limit_seconds, verbose, simplified_resolve=True)
            if solution_without_energy is not None:
                print('+++ Success; using solution without energy as initial solution +++')
                from drtOrtools import solution_by_requests
                solution_without_energy = solution_by_requests(solution_without_energy, data)
                data.include_charging = True
                data.initial_routes = solution_without_energy
                solution = main(data, time_limit_seconds, verbose, simplified_resolve=True)
                return solution
            else:
                print('+++ Cannot find a solution even ignoring energy. +++')
        else:
            print('***** Routing optimization failed in some way *****')
    # Todo Philipp: Clean this up or remove it
    # if solution and data.include_charging:
    #     for node in data.starts:
    #         index = manager.NodeToIndex(node)
    #         print(f'Initial charge (in node {node}): {energy_dimension.CumulVar(index)}')
    #     for co in data.charging_opportunities:
    #         index = manager.NodeToIndex(co.node)
    #     for node, energy in enumerate(data['available_energy']):
    #         if True:# Todo Philipp: Activate energy_dimension printing only when Energy is active
    #             index = manager.NodeToIndex(node)
    #             if node in data['starts']:
    #                 print(f'Initial charge (in node {node}): {energy}')
    #             elif energy > 0:
    #                 planned_charging = solution.Value(energy_dimension.CumulVar(index))
    #                 suffix = '############ MAX CHARGE EXCEEDED ############' if planned_charging > 70000 else ''
    #                 print(f'Node {node}: Planned charging {planned_charging} / {energy} {suffix}')

    if solution:
        return get_solution(data, manager, routing, solution, verbose)
    else:
        if verbose:
            print('There is no solution.')
    return None
