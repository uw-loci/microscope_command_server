"""Autofocus test and benchmark command handlers.

Handles autofocus testing and benchmarking commands:
TESTAF, TESTADAF, TESTAFV, AFBENCH
"""

import socket
import logging

from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)


def handle_testaf(conn, client, hardware, settings, **kwargs):
    """Test standard autofocus at the current position.

    Reads a message with --yaml, --output, --objective flags,
    then runs test_standard_autofocus_at_current_position.

    Response: SUCCESS:<plot_path>|<initial_z>:<final_z>:<z_shift>
              or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested autofocus test", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read autofocus test message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse the message
    params = {}
    flags = ["--yaml", "--output", "--objective"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1 :]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()
            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--objective":
                params["objective"] = value

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path", "objective"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    # Send immediate acknowledgment to prevent client timeout
    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for standard autofocus test")

        # Execute STANDARD autofocus test
        from microscope_control.autofocus.test import (
            test_standard_autofocus_at_current_position,
        )

        result = test_standard_autofocus_at_current_position(
            hardware=hardware,
            config_manager=config_manager,
            yaml_file_path=params["yaml_file_path"],
            output_folder_path=params["output_folder_path"],
            objective=params["objective"],
            logger=logger,
        )

        if result["success"]:
            # Format result as: SUCCESS:plot_path|initial_z:final_z:z_shift
            result_data = (
                f"{result['initial_z']:.2f}:" f"{result['final_z']:.2f}:" f"{result['z_shift']:.2f}"
            )
            response = f"SUCCESS:{result['plot_path']}|{result_data}".encode()
            conn.sendall(response)
            logger.info("Autofocus test completed: %s", result["message"])
        else:
            response = f"FAILED:{result['message']}".encode()
            conn.sendall(response)
            logger.error("Autofocus test failed: %s", result["message"])

    except Exception as e:
        logger.error("Autofocus test failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


def handle_testadaf(conn, client, hardware, settings, **kwargs):
    """Test adaptive autofocus at the current position.

    Reads a message with --yaml, --output, --objective flags,
    then runs test_adaptive_autofocus_at_current_position.

    Response: SUCCESS:<message>|<initial_z>:<final_z>:<z_shift>
              or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested adaptive autofocus test", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read adaptive autofocus test message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse the message
    params = {}
    flags = ["--yaml", "--output", "--objective"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1 :]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()
            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--objective":
                params["objective"] = value

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path", "objective"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    # Send immediate acknowledgment to prevent client timeout
    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for adaptive autofocus test")

        # Execute ADAPTIVE autofocus test
        from microscope_control.autofocus.test import (
            test_adaptive_autofocus_at_current_position,
        )

        result = test_adaptive_autofocus_at_current_position(
            hardware=hardware,
            config_manager=config_manager,
            yaml_file_path=params["yaml_file_path"],
            output_folder_path=params["output_folder_path"],
            objective=params["objective"],
            logger=logger,
        )

        if result["success"]:
            # Format result as: SUCCESS:message|initial_z:final_z:z_shift
            result_data = (
                f"{result['initial_z']:.2f}:" f"{result['final_z']:.2f}:" f"{result['z_shift']:.2f}"
            )
            response = f"SUCCESS:{result['message']}|{result_data}".encode()
            conn.sendall(response)
            logger.info("Adaptive autofocus test completed: %s", result["message"])
        else:
            response = f"FAILED:{result['message']}".encode()
            conn.sendall(response)
            logger.error("Adaptive autofocus test failed: %s", result["message"])

    except Exception as e:
        logger.error("Adaptive autofocus test failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


def handle_testafv(conn, client, hardware, settings, **kwargs):
    """Test autofocus validation (sweep + recovery from defocus).

    Reads a message with --yaml, --output, --objective flags,
    then runs test_autofocus_validation.

    Response: SUCCESS:<json_result> or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested autofocus validation test", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read autofocus validation message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse the message (same flags as TESTAF)
    params = {}
    flags = ["--yaml", "--output", "--objective"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1 :]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()
            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--objective":
                params["objective"] = value

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path", "objective"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    # Send immediate acknowledgment
    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for autofocus validation test")

        # Execute autofocus validation test
        from microscope_control.autofocus.test import (
            test_autofocus_validation,
        )

        result = test_autofocus_validation(
            hardware=hardware,
            config_manager=config_manager,
            yaml_file_path=params["yaml_file_path"],
            objective=params["objective"],
            logger=logger,
        )

        if result["success"]:
            # Format: SUCCESS:JSON-encoded result
            import json

            result_json = json.dumps(result)
            response = f"SUCCESS:{result_json}".encode()
            conn.sendall(response)
            logger.info(
                "Autofocus validation test completed: " "sweep_delta=%sum, recovery_delta=%sum",
                result["sweep_delta_um"],
                result["recovery_delta_um"],
            )
        else:
            response = f"FAILED:{result['message']}".encode()
            conn.sendall(response)
            logger.error("Autofocus validation test failed: %s", result["message"])

    except Exception as e:
        logger.error("Autofocus validation test failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


def handle_afbench(conn, client, hardware, settings, **kwargs):
    """Run autofocus parameter benchmark.

    Reads a message with --reference_z, --output, --distances,
    --quick, --objective flags, then runs the benchmark with
    progress updates sent back through the socket.

    Response: SUCCESS:<summary>|<results_dir> or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested autofocus benchmark", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read benchmark message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse the message
    params = {}
    flags = ["--reference_z", "--output", "--distances", "--quick", "--objective"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1 :]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()

            if flag == "--reference_z":
                params["reference_z"] = float(value)
            elif flag == "--output":
                params["output_folder"] = value
            elif flag == "--distances":
                # Parse comma-separated distances
                params["test_distances"] = [float(d.strip()) for d in value.split(",")]
            elif flag == "--quick":
                params["quick_mode"] = value.lower() in ("true", "1", "yes")
            elif flag == "--objective":
                params["objective"] = value

    # Validate required parameters
    required = ["reference_z", "output_folder"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    # Send immediate acknowledgment
    try:
        ack_response = f"STARTED:{params['output_folder']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for autofocus benchmark")

        # Create progress callback that sends socket updates
        # This keeps the connection alive during long benchmarks
        # Format: PROGRESS:current:total:message (consistent with PPMBIREF)
        def send_progress(current, total, status_msg):
            """Send progress update to keep connection alive."""
            try:
                progress_msg = f"PROGRESS:{current}:{total}:{status_msg}"
                conn.sendall(progress_msg.encode())
            except Exception as e:
                logger.warning("Failed to send progress update: %s", e)

        # Execute benchmark with progress callback
        from microscope_control.autofocus.benchmark import (
            run_autofocus_benchmark_from_server,
        )

        result = run_autofocus_benchmark_from_server(
            hardware=hardware,
            config_manager=config_manager,
            reference_z=params["reference_z"],
            output_folder=params["output_folder"],
            test_distances=params.get("test_distances"),
            quick_mode=params.get("quick_mode", False),
            objective=params.get("objective"),
            logger=logger,
            progress_callback=send_progress,
        )

        # Check for safety violation
        if result.get("safety_violation"):
            error_msg = result.get("error", "Safety limit exceeded")
            response = f"FAILED:SAFETY:{error_msg}".encode()
            conn.sendall(response)
            logger.error("Autofocus benchmark SAFETY VIOLATION: %s", error_msg)
        else:
            # Format response
            success_rate = result.get("success_rate", 0)
            total_trials = result.get("total_trials", 0)
            results_dir = result.get("results_directory", "")

            response = (
                f"SUCCESS:Benchmark complete. {total_trials} trials, "
                f"{success_rate:.1%} success rate|{results_dir}"
            ).encode()
            conn.sendall(response)
            logger.info("Autofocus benchmark completed: %d trials", total_trials)

    except Exception as e:
        logger.error("Autofocus benchmark failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)
