package smos.application.registerManagement;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Date;
import java.util.Iterator;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import smos.Environment;
import smos.bean.Absence;
import smos.bean.Classroom;
import smos.bean.Delay;
import smos.bean.RegisterLine;
import smos.bean.User;
import smos.bean.UserListItem;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.ManagerRegister;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;
import smos.utility.Utility;

public class ServletUpdateRegister extends HttpServlet {

	private static final long serialVersionUID = 5966298318913522686L;
	
	protected void doGet(HttpServletRequest pRequest,
			HttpServletResponse pResponse) {
		String gotoPage = "./registerManagement/showClassroomList.jsp";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		
		//Boolean variable used to check whether the student has an absence or not
		boolean flag = false;
		
		//Collection used for log storage of a particular date
		Collection<RegisterLine> register = null;
		//Iterator needed to scroll through the collection
		Iterator itRegister = null;
		//Temporary variable needed to read information from the collection
		RegisterLine tmpRegisterLine = null;
		//Temporary variable needed to read information from the collection
		UserListItem student = null;
		
		//Temporary variable needed to insert new absences
		Absence tmpAbsence = null;
		
		//Temporary variable needed to insert new delays
		Delay tmpDelay = null;
		
		//Manager classes required for processing
		ManagerUser managerUser = ManagerUser.getInstance();
		ManagerRegister managerRegister = ManagerRegister.getInstance();
		
		//Variables necessary for storing the data coming from the request
		String[] absences = null; //Memorize absent pupils
		String[] delays = null; //Memorize laggards
		
		//Retrieve the user logged in from the session
		User loggedUser = (User) session.getAttribute("loggedUser");
		//I verify that the logged in user has the necessary permissions
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			}

			if (!managerUser.isAdministrator(loggedUser)) {
				errorMessage = "The logged in User does not have access to the "
						+ "functionality'!";
				gotoPage = "./error.jsp";
				return;
			}
		
		//Retrieving the parameters from pRequest
		Date date = Utility.String2Date(pRequest.getParameter("date"));
		absences = pRequest.getParameterValues("absences");
		delays = pRequest.getParameterValues("delays");
		
		//Retrieve the classroom object from the session
		Classroom classroom = ((Classroom) session.getAttribute("classroom"));
		
		/*I call the managerRegister method to retrieve the information from the db
		 * concerning the register of a class on a particular date (Absences, Delays)
		 */
		register = managerRegister.getRegisterByClassIDAndDate(classroom.getIdClassroom(),date);
			
		if (register != null){
			itRegister = register.iterator();
		}
		
		if (itRegister != null){
			while(itRegister.hasNext()){
				tmpRegisterLine = (RegisterLine) itRegister.next();
				//I pick up the student to whom the register line refers
				student = tmpRegisterLine.getStudent();
				
				//I check if an absence has been entered for the student or not
				if (absences != null){
					for (int i=0; i<absences.length; i++){
						if (Integer.valueOf(absences[i]) == student.getId()){
							flag = true;
							if (!managerRegister.hasAbsence(tmpRegisterLine)){
								tmpAbsence = new Absence();
								tmpAbsence.setAcademicYear(classroom.getAcademicYear());
								tmpAbsence.setDateAbsence(date);
								tmpAbsence.setIdJustify(0);
								tmpAbsence.setIdUser(student.getId());
								managerRegister.insertAbsence(tmpAbsence);
							}
						}
						
					}
					if (!flag){
						if (managerRegister.hasAbsence(tmpRegisterLine)){
							managerRegister.deleteAbsence(tmpRegisterLine.getAbsence());
						}
					}
				} else {
					if (managerRegister.hasAbsence(tmpRegisterLine)){
						managerRegister.deleteAbsence(tmpRegisterLine.getAbsence());
					}
				}
				flag = false;
				
				//I check if a delay has been entered for the student or not
				if (delays != null){
					for (int i=0; i<delays.length; i++){
						if (Integer.valueOf(delays[i]) == student.getId()){
							flag = true;
							if (!managerRegister.hasDelay(tmpRegisterLine)){
								tmpDelay = new Delay();
								tmpDelay.setAcademicYear(classroom.getAcademicYear());
								tmpDelay.setDateDelay(date);
								tmpDelay.setIdUser(student.getId());
								tmpDelay.setTimeDelay(pRequest.getParameter("hour_" + student.getId()));
								managerRegister.insertDelay(tmpDelay);
							} else {
								tmpDelay = tmpRegisterLine.getDelay();
								tmpDelay.setTimeDelay(pRequest.getParameter("hour_" + student.getId()));
								managerRegister.updateDelay(tmpDelay);
							}
							
						}
						
					}
					if (!flag){
						if (managerRegister.hasDelay(tmpRegisterLine)){
							managerRegister.deleteDelay(tmpRegisterLine.getDelay());
						}
					}
				} else {
					if (managerRegister.hasDelay(tmpRegisterLine)){
						managerRegister.deleteDelay(tmpRegisterLine.getDelay());
					}
				}
				flag = false;
			}
		}
			
			

		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		} catch (SQLException sqlException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ sqlException.getMessage();
			gotoPage = "./error.jsp";
			sqlException.printStackTrace();
		} catch (EntityNotFoundException entityNotFoundException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (ConnectionException connectionException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ connectionException.getMessage();
			gotoPage = "./error.jsp";
			connectionException.printStackTrace();
		} catch (InvalidValueException e) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + e.getMessage();
			gotoPage = "./error.jsp";
			e.printStackTrace();
		} catch (MandatoryFieldException e) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + e.getMessage();
			gotoPage = "./error.jsp";
			e.printStackTrace();
		}
		
		
		session.setAttribute("errorMessage", errorMessage);
		try {
			pResponse.sendRedirect(gotoPage);
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}

	}

	protected void doPost(HttpServletRequest pRequest,
			HttpServletResponse pResponse) {
		this.doGet(pRequest, pResponse);
	}

}
