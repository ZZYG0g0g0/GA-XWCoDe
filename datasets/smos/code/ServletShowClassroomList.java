package smos.application.classroomManagement;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collection;


import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import smos.Environment;
import smos.bean.Classroom;
import smos.bean.User;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.storage.ManagerClassroom;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;

public class ServletShowClassroomList extends HttpServlet {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8468821050771730936L;

	/**
	 * Definition of the doGet method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doGet(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		String gotoPage = "./persistentDataManagement/classroomManagement/showClassroomList.jsp";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		Collection<Classroom> classroomList = null;
		ManagerUser managerUser = ManagerUser.getInstance();
		ManagerClassroom managerClassroom= ManagerClassroom.getInstance();
		User loggedUser = (User) session.getAttribute("loggedUser");
			
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			} 
			if (!managerUser.isAdministrator(loggedUser)) {
				errorMessage =  "The logged in User does not have access to the " +
						"functionality'!";
				gotoPage = "./error.jsp";
			} 
			
			//take the academic year from the session
			
			classroomList = managerClassroom.getClassroomsByAcademicYear(Integer.valueOf(pRequest.getParameter("academicYear")));
			
					
			session.setAttribute("classroomList", classroomList);
			pResponse.sendRedirect(gotoPage);
			return; 
			
		} catch (SQLException sqlException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + sqlException.getMessage();
			gotoPage = "./error.jsp";
			sqlException.printStackTrace();
		} catch (EntityNotFoundException entityNotFoundException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (ConnectionException connectionException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + connectionException.getMessage();
			gotoPage = "./error.jsp";
			connectionException.printStackTrace();
		}catch (IOException ioException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}catch (InvalidValueException invalidValueException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + invalidValueException.getMessage();
			gotoPage = "./error.jsp";
			invalidValueException.printStackTrace();
		}
		
		pRequest.getSession().setAttribute("errorMessage", errorMessage);
		try {
			pResponse.sendRedirect(gotoPage);
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}
	}

	/**
	 * Definition of the doPost method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doPost(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		this.doGet(pRequest, pResponse);
	}

}
